# HAI transfer 001 monitoring note

The GitHub Actions watch was bounded and did not loop forever. It queried status once while the Kaggle run was queued, slept for 900 seconds, then queried again and observed the terminal error. The Kaggle log shows the scientific process had already failed after roughly 104 seconds.

Repair 002 changes only monitoring responsiveness: `30`, `60`, `120`, and `300` second startup delays precede the existing `900` second steady interval. The watch remains capped at 24 status calls and 210 minutes and does not restart failed compute.
