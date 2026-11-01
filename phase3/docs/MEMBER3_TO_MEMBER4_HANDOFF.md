# Member 3 to Member 4 adapter handoff

This handoff is valid only after Member 3B records an accepted training-acceptance record. Provide: selected adapter/checkpoint identity and SHA-256; pinned `popo28/Venus-Q-Stage1` revision; Qwen source commit and reviewed patch hash; LoRA target modules/rank/alpha/dropout/bias/modules-to-save; BF16 load policy; exact BridgeTrain/B1 prompt identity; clean-process reload evidence; formal config hash; training acceptance hash; and the adapter's private runtime path through the team's approved non-Git channel.

Do not include model weights, adapter bytes, private image paths, credentials, or raw logs in Git. Member 4 owns inference loading and Condition C integration; Member 3 does not modify frozen A/B0/B1 behavior.
