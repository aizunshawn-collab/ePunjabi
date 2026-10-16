import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_pa_hybrid_ctc_rnnt_large")
model.freeze()
model.cur_decoder = "rnnt"
text = model.transcribe(["punjabi_test_16k.wav"], batch_size=1, language_id="pa")[0]
print("RNNT_RESULT:", text)
model.cur_decoder = "ctc"
text2 = model.transcribe(["punjabi_test_16k.wav"], batch_size=1, logprobs=False, language_id="pa")[0]
print("CTC_RESULT:", text2)
