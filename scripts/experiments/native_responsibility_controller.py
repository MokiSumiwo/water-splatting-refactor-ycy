"""Serializable event controller for the native B/N training worker.

No optimizer updates are performed here: every compensation step belongs to
one iteration of the worker. References are detached native renders.
"""
from __future__ import annotations

import copy
from pathlib import Path
import torch
import torch.nn.functional as F


def load_event_asset(path):
    """Reconstruct either a full snapshot or its lossless event delta."""
    path = Path(path)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('base'):
        base = load_event_asset(path.parent / payload['base'])
        payload['model'] = {**base['model'], **payload['model']}
    if payload.get('state_base'):
        base = load_event_asset(path.parent / payload['state_base'])
        payload['state'] = {**base['state'], **payload['state']}
    return payload


class ResponsibilityController:
    def __init__(self, worker, branch, records, arm, output, smoke=False, state=None):
        self.w = worker
        self.branch = branch
        self.model = branch.pipeline.model
        self.records = records
        self.arm = arm
        self.output = Path(output)
        self.reference_step = 4 if smoke else 10000
        self.block = 12 if smoke else 200
        self.inner = 4 if smoke else 32
        self.events = 1 if smoke else 20
        self.smoke = smoke
        ordered = sorted(range(len(records)), key=lambda i: str(records[i][1]))
        n = min(8, len(ordered))
        if n < 2:
            raise ValueError('responsibility training needs at least two train cameras')
        selected = [ordered[round(i * (len(ordered)-1)/(n-1))] for i in range(n)]
        fit_count = min(6, max(1, round(n * .75)))
        self.state = state or dict(version=1, formal_step=0, event_index=0,
            phase='native', rho_planned=0., rho_effective=0., rho_last_good=0.,
            paused=False, fit_indices=selected[:fit_count], monitor_indices=selected[fit_count:],
            cap=None, optimizer_calls={}, history=[], inner_offset=0)
        self.state = self.to_device(self.state, self.model.device)
        # Older checkpoints used the worker's hashing normalizer, which
        # stringified camera IDs. Restore only the known camera-index banks.
        for name in ('pixels', 'masks', 'counterfactual'):
            if name in self.state:
                self.state[name] = {int(k): v for k, v in self.state[name].items()}

    @staticmethod
    def to_cpu(value):
        """Checkpoint values losslessly; camera IDs must remain integers."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: ResponsibilityController.to_cpu(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(ResponsibilityController.to_cpu(v) for v in value)
        return value

    @staticmethod
    def to_device(value, device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {k: ResponsibilityController.to_device(v, device) for k,v in value.items()}
        if isinstance(value, list):
            return [ResponsibilityController.to_device(v, device) for v in value]
        return value

    def phase(self, step):
        offset = step-self.reference_step-1
        if offset < 0:
            return 'native', 0, 0
        if offset >= self.block*self.events:
            return 'stabilize', self.events, 0
        return ('m1_only' if offset%self.block < self.inner else 'normal',
                offset//self.block+1, offset%self.block)

    def camera(self, step, default):
        phase, _, offset = self.phase(step)
        ids = self.state['fit_indices']
        return ids[offset%len(ids)] if phase == 'm1_only' else default

    def render(self, index):
        camera = self.records[index][2].to(self.model.device)
        return self.model.get_outputs(camera)

    def raw(self, outputs):
        # These are the nine channels immediately after OCMC, before the
        # medium activations: raw_base + the projected camera delta.
        return outputs['camera_medium_raw_base'].float() + outputs['camera_medium_delta_projected_raw'].float()

    def masks(self, outputs, batch):
        gt = self.w._gt(self.model, batch, outputs['background']).detach().float()
        depth = outputs['depth'].detach().reshape(gt.shape[:2])
        # Fixed far-field observation window, not a water segmentation.
        valid = torch.isfinite(depth) & (depth > 0)
        cutoff = torch.quantile(depth[valid], .65) if valid.any() else depth.new_tensor(float('inf'))
        roi = valid & (depth >= cutoff)
        gray = gt.mean(-1)
        edge = torch.zeros_like(gray)
        edge[1:] += (gray[1:]-gray[:-1]).abs()
        edge[:,1:] += (gray[:,1:]-gray[:,:-1]).abs()
        structure = edge > torch.quantile(edge.reshape(-1), .85)
        return roi, structure & ~roi, gt

    def pixel_bank(self, roi, structure):
        masks = [roi, structure & ~roi, ~(roi | structure)]
        quotas = [4096,2048,2048]
        result = []
        for mask, quota in zip(masks, quotas):
            ids = torch.where(mask.reshape(-1))[0]
            if ids.numel():
                # Uniform deterministic positions, no training RNG consumed.
                pos = torch.linspace(0, ids.numel()-1, min(quota,ids.numel()), device=ids.device).long()
                result.append(ids[pos])
            else:
                result.append(ids)
        return result

    @torch.no_grad()
    def establish_reference(self):
        m, w = self.model, self.w
        n = int(m.num_points)
        contribution = torch.zeros(n, device=m.device)
        protected = torch.zeros(n, device=m.device)
        footprint = torch.zeros(n, device=m.device)
        pixels, scales, masks = {}, [], {}
        for index in self.state['fit_indices'] + self.state['monitor_indices']:
            out = self.render(index)
            batch = w._batch_to_device(self.records[index][3].copy(), m.device)
            roi, edge, gt = self.masks(out, batch)
            masks[index] = dict(roi=roi, structure=edge, reason='training-view depth upper 35 percent; observed edge upper 15 percent')
            pixels[index] = self.pixel_bank(roi,edge)
            raw = self.raw(out).reshape(-1,9)
            scales.append(raw[torch.cat(pixels[index])])
            geom = w.FULL._project_for_camera(m, self.records[index][2].to(m.device))
            colors = out['gaussian_view_rgb'].detach().float()
            selected = torch.arange(n,device=m.device)
            z = torch.zeros_like(gt[...,0])
            stats, *_ = w.FULL._responsibility_forward(w._responsibility_extension(), geom, colors,
                torch.sigmoid(m.opacities.detach()).reshape(-1), out['medium_attn'].detach().float(),
                selected, roi[...,None].expand_as(gt).float().contiguous(), edge.float(), z)
            # stats[1:4] = sum T*alpha*ROI; colors supply additive clear RGB.
            contribution += (stats[:,1:4]*colors).mean(-1)
            footprint += stats[:,0]
            protected += stats[:,10]
        # Conservatively protect a whole Gaussian when it contributes to
        # observed structure. No missing-track => water assumption is made.
        safe = protected <= 1e-6
        rank = torch.argsort(contribution * safe, descending=True)
        rank = rank[(contribution[rank] > 0) & safe[rank]][:4096]
        total = contribution.sum()
        if rank.numel():
            prefix = contribution[rank].cumsum(0)
            count = min(rank.numel(), int(torch.searchsorted(prefix,total*.5).item())+1)
            rank = rank[:count]
        mask = torch.zeros(n,dtype=torch.bool,device=m.device)
        mask[rank] = True
        cap = w._opacity_cap_state(m, mask)
        cap['gaussian_ids'] = torch.arange(n,device=m.device,dtype=torch.long)
        cap['footprint_weights'] = footprint
        cap['target_contribution'] = contribution
        self.state.update(cap=cap, pixels=pixels, masks=masks,
            coverage=float(contribution[mask].sum()/total.clamp_min(1e-12)),
            protected_coverage=float(contribution[~safe].sum()/total.clamp_min(1e-12)))
        raw = torch.cat(scales)
        median = raw.median(0).values
        # Robust 10k per-channel MAD; explicit raw-channel floor.
        self.state['trust_scale'] = (1.4826*(raw-median).abs().median(0).values).clamp_min(.01)
        self.state['reference_denominator'] = float((footprint[mask]*cap['reference_tau'][mask]).sum())
        self.state['reference_population'] = n
        self.save_assets('reference')

    def snapshot(self):
        return {k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}

    def save_assets(self, name, extra=None):
        path = self.output/'events'/f'{name}.pt'
        path.parent.mkdir(parents=True,exist_ok=True)
        snapshot = self.snapshot()
        state = self.to_cpu(self.state)
        base = None
        state_base = None
        if name != 'reference' and not self.smoke:
            # Geometry is frozen at the reference. SH and opacity are shared
            # within each event's inner window. Retain exact tensors once.
            base = 'reference.pt'
            prefixes = ('gauss_params.features_', 'gauss_params.opacities',
                        'medium_mlp.', 'direction_encoding.')
            if name.endswith('_capped'):
                base = name.replace('_capped', '_S0') + '.pt'
                prefixes = ('gauss_params.opacities',)
            elif name.endswith('_M1'):
                base = name.replace('_M1', '_capped') + '.pt'
                prefixes = ('medium_mlp.', 'direction_encoding.')
            snapshot = {k:v for k,v in snapshot.items() if k.startswith(prefixes)}
            if self.state['cap']['gaussian_ids'].numel() == self.state['reference_population']:
                state_base = 'reference.pt'
                for key in ('cap','pixels','masks','trust_scale'):
                    state.pop(key, None)
        torch.save(dict(model=snapshot, state=state, base=base,
                        state_base=state_base, extra=extra),path)

    @torch.no_grad()
    def monitor(self):
        values = []
        for index in self.state['monitor_indices']:
            out = self.render(index)
            batch = self.w._batch_to_device(self.records[index][3].copy(),self.model.device)
            gt = self.w._gt(self.model,batch,out['background']).float().clamp(0,1)
            pred = out['pred_image'].float().clamp(0,1)
            mse = (pred-gt).square().mean(-1)
            mask = self.state['masks'][index]
            roi, edge = mask['roi'], mask['structure']
            coverage = out['accumulation'].reshape(mse.shape)
            residual = pred-gt
            gradient = (residual[1:]-residual[:-1]).abs().mean(-1)
            values.append(dict(psnr=float(-10*torch.log10(mse.mean().clamp_min(1e-12))),
                roi_mse=float(mse[roi].mean()) if roi.any() else None,
                coverage=float(coverage[edge].mean()) if edge.any() else None,
                edge_error=float(gradient[edge[1:]].mean()) if edge[1:].any() else None))
        result = {}
        for key in values[0]:
            defined = [v[key] for v in values if v[key] is not None]
            result[key] = sum(defined)/len(defined) if defined else None
        return result

    @staticmethod
    def alarms(pre, now):
        flags = []
        if pre['psnr']-now['psnr'] > .20: flags.append('psnr')
        if pre['roi_mse'] is not None and now['roi_mse']-pre['roi_mse'] > max(.1*pre['roi_mse'],1e-5): flags.append('roi_mse')
        if (pre['coverage'] is not None and now['coverage'] is not None and
            pre['edge_error'] is not None and now['edge_error'] is not None and
            pre['coverage']-now['coverage'] > .03 and now['edge_error'] > 1.1*pre['edge_error']): flags.append('structure')
        return flags

    @torch.no_grad()
    def begin_event(self, event):
        self.state['event_index'] = event
        self.state['rho_planned'] = .75*event/20 if self.arm == 'N' else 0.
        self.state['event_pre'] = self.monitor()
        ids = self.state['fit_indices']
        before = {}
        clear_before = {}
        for i in ids:
            out = self.render(i)
            pix = torch.cat(self.state['pixels'][i])
            clear_before[i] = out['rgb_clear'].reshape(-1,3)[pix].detach().float().clone()
            before[i] = dict(I0=out['pred_image'].reshape(-1,3)[pix].detach().clone(),
                raw0=self.raw(out).reshape(-1,9)[pix].detach().clone())
        self.state['counterfactual'] = before
        self.save_assets(f'event_{event:02d}_S0')
        if not self.state['paused']:
            self.state['rho_effective'] = self.state['rho_planned']
        opacity_before = torch.sigmoid(self.model.opacities.detach().float()).reshape(-1).clone()
        projection = self.project()
        cap = self.state['cap']
        weights = cap['footprint_weights'] * cap['candidate_mask']
        opacity_after = torch.sigmoid(self.model.opacities.detach().float()).reshape(-1)
        tau_before = -torch.log1p(-opacity_before.clamp(max=1-1e-7))
        tau_after = -torch.log1p(-opacity_after.clamp(max=1-1e-7))
        self.state['event_projection'] = dict(projection,
            weighted_tau_before=float((weights*tau_before).sum()),
            weighted_tau_after=float((weights*tau_after).sum()),
            weighted_tau_reference=self.state['reference_denominator'],
            actual_candidate_hits=int(((opacity_before-opacity_after>1e-7)&cap['candidate_mask']).sum()))
        clear_change, underwater_change = [], []
        for i in ids:
            out = self.render(i)
            pix = torch.cat(self.state['pixels'][i])
            before[i]['Ic'] = out['pred_image'].reshape(-1,3)[pix].detach().clone()
            before[i]['b'] = before[i]['I0']-before[i]['Ic']
            underwater_change.append(float(before[i]['b'].float().abs().mean()))
            clear_change.append(float((out['rgb_clear'].reshape(-1,3)[pix].float()-clear_before[i]).abs().mean()))
        self.state['event_projection'].update(clear_mean_abs=sum(clear_change)/len(ids),
            underwater_mean_abs=sum(underwater_change)/len(ids))
        self.save_assets(f'event_{event:02d}_capped')

    def inner_loss(self, outputs, index):
        bank = self.state['counterfactual'][index]
        strata = self.state['pixels'][index]
        pix = torch.cat(strata)
        pred = outputs['pred_image'].reshape(-1,3)[pix]
        residual = (pred-bank['Ic'])-bank['b']
        terms, weights, offset = [], [], 0
        for ids, weight in zip(strata,[.5,.25,.25]):
            if ids.numel():
                terms.append(F.huber_loss(residual[offset:offset+ids.numel()],torch.zeros_like(residual[offset:offset+ids.numel()]),delta=.01))
                weights.append(weight)
            offset += ids.numel()
        transfer = sum(t*weight for t,weight in zip(terms,weights))/sum(weights)
        raw = self.raw(outputs).reshape(-1,9)[pix]
        trust = ((raw-bank['raw0'])/self.state['trust_scale']).square().mean()
        return .25*transfer + .01*trust, dict(transfer=float(transfer.detach()),trust=float(trust.detach()))

    def project(self):
        if self.state['cap'] is None: return {}
        stats = self.w._project_opacity_(self.model,self.branch.optimizers,self.state['cap'],self.state['rho_effective'])
        if stats['max_opacity_violation'] > 1e-6:
            raise RuntimeError(f'opacity cap violation: {stats}')
        return stats

    def synchronize_cull(self, deleted):
        """Apply the native cull mapping without touching medium optimizers."""
        cap = self.state['cap']
        if cap is None or deleted.numel() != cap['gaussian_ids'].numel():
            raise RuntimeError('cull mapping does not match reference IDs')
        removed = cap['gaussian_ids'][deleted].detach().cpu().tolist()
        for name, value in list(cap.items()):
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == deleted.numel():
                cap[name] = value[~deleted]
        self.state.setdefault('cull_history', []).append(dict(step=self.state['formal_step'], removed_ids=removed))
        self.project()

    def before(self, step):
        phase, event, offset = self.phase(step)
        if phase == 'm1_only' and offset == 0:
            self.begin_event(event)
        self.state.update(phase=phase, event_index=event, inner_offset=offset if phase == 'm1_only' else 0)
        return phase

    def after(self, step):
        self.state['formal_step'] = step
        if step == self.reference_step:
            self.establish_reference()
        stats = self.project()
        phase, event, offset = self.phase(step)
        if event and offset == self.inner-1 and phase == 'm1_only':
            self.state['post_inner'] = self.monitor()
            self.state['inner_alarms'] = self.alarms(self.state['event_pre'],self.state['post_inner'])
            self.save_assets(f'event_{event:02d}_M1')
        if event and offset == self.block-1 and step <= self.reference_step+self.block*self.events:
            end = self.monitor()
            alarms = self.alarms(self.state['event_pre'],end)
            persistent = sorted(set(alarms)&set(self.state.get('inner_alarms',[])))
            record = dict(event=event,step=step,pre=self.state['event_pre'],
                post_inner=self.state['post_inner'],end=end,alarms=persistent,
                projection=self.state.get('event_projection',{}),coverage=self.state.get('coverage'),
                rho_planned=self.state['rho_planned'],rho_applied=self.state['rho_effective'])
            self.state['history'].append(record)
            if persistent:
                self.save_assets(f'event_{event:02d}_alarm')
                self.state['paused'] = True
                self.state['rho_effective'] = 0. if 'structure' in persistent else self.state['rho_last_good']
            elif not self.state['paused']:
                self.state['rho_last_good'] = self.state['rho_effective']
            self.w._write_json(self.output/'event_history.json',self.state['history'])
        return stats
