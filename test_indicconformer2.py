import nemo.collections.asr as nemo_asr
nemo_path = r"C:\Users\aizun\.cache\torch\NeMo\NeMo_3.0.0\hf_hub_cache\ai4bharat\indicconformer_stt_pa_hybrid_ctc_rnnt_large\eee0706c7ee9fc394a11db23233df803\models--ai4bharat--indicconformer_stt_pa_hybrid_ctc_rnnt_large\indicconformer_stt_pa_hybrid_rnnt_large.nemo"
import os
print("exists:", os.path.exists(nemo_path))
model = nemo_asr.models.ASRModel.restore_from(restore_path=nemo_path)
model.freeze()
model.cur_decoder = "rnnt"
text = model.transcribe(["punjabi_test_16k.wav"], batch_size=1, language_id="pa")[0]
print("RNNT_RESULT:", text)
model.cur_decoder = "ctc"
text2 = model.transcribe(["punjabi_test_16k.wav"], batch_size=1, logprobs=False, language_id="pa")[0]
print("CTC_RESULT:", text2)
