# ogvoice-runpod-runner

RunPod serverless worker used by OG Voice for both training and inference.

## RunPod worker (`handler.py`)

Supported modes:

- `mode=train` for voice cloning training
- `mode=infer` for AI cover inference (phase-1 pipeline)

Inference input expects:

- `inputAudioKey`
- `modelArtifactKey`
- `outKey`
- `config`
- optional `stemKeys`

Inference output returns:

- `outputKey`
- `outputBytes`
- `stemKeys`
