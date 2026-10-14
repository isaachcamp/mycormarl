# Research execution boundary

`mycormarl.static_rollout.evaluate_static_policy` executes supplied physical actions in a supplied environment; its batched counterpart accepts environments sharing dynamics and organism traits. These functions retain numerical resource-accounting checks and return trajectory facts. Named manifests, treatment matrices, scientific stopping rules, and result identities belong to REDbyRBD.

PPO, policy artifacts, checkpoint evaluation, fixed-allocation action conversion, and RNG contracts remain core. No research installation is needed for those facilities. The legacy study and named-baseline entry points forward lazily to the separately installed REDbyRBD package and fail clearly when it is absent. The old static manifest adapter temporarily remains for scripts scheduled to migrate in P3 and is removed at final cutover.

Historical source is preserved on the original feature branch. Scientific outputs retain their generating identities; forwarding an entry point does not make old results compatible with the new cross-repository result contract.
