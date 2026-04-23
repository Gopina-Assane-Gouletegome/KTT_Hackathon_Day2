# process_log.md — T2.1 · Crop Disease Classifier

## LLM / Tool Use Declaration

I used Claude (Anthropic) as an AI assistant during this hackathon. Specifically:
- Generated boilerplate code for training pipeline, FastAPI service, and Dockerfile
- Reviewed and corrected generated code before running
- All model design decisions, hyperparameter choices, and evaluation were performed by me
- The USSD fallback document structure was AI-assisted; content was verified against real Rwanda RAB extension data

---

## Hour-by-Hour Timeline

### Hour 1 (09:00 – 10:00)
- [ ] Read and understood challenge brief
- [ ] Set up Colab environment, installed dependencies
- [ ] Downloaded/generated dataset, verified class distributions
- [ ] Started Phase 1 training (head only)

### Hour 2 (10:00 – 11:00)
- [ ] Monitored training, adjusted hyperparameters if val accuracy stalled
- [ ] Phase 2 fine-tuning started
- [ ] Evaluated on clean test set — confirmed macro-F1 ≥ 0.80
- [ ] Exported ONNX and TFLite INT8, verified file sizes < 10 MB

### Hour 3 (11:00 – 12:00)
- [ ] Ran robustness evaluation on test_field.zip
- [ ] Launched FastAPI service locally, tested with curl
- [ ] Built and tested Docker container
- [ ] Wrote ussd_fallback.md (reviewed AI draft, added local specifics)

### Hour 4 (12:00 – 13:00)
- [ ] Final README polish
- [ ] Uploaded model to Hugging Face Hub + wrote model card
- [ ] Recorded 4-minute demo video
- [ ] Pushed to GitHub, verified 2-command reproduction
- [ ] Submitted

---

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| MobileNetV3-Small over EfficientNet-B0 | Smaller parameter count (~2.5M vs ~5.3M); fits under 10MB after quantization |
| TF/Keras over PyTorch | Faster TFLite INT8 export pipeline with native converter |
| FastAPI over Flask | Auto-generated OpenAPI docs; Pydantic validation; async support |
| Village Agent relay model | Rwanda has 4,500+ trained VAs — more scalable than kiosk model |

---

*Fill in actual times and notes as you work through the challenge.*