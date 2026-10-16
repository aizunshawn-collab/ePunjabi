import sys
sys.path.insert(0, '/app')
from indicconformer_nemo_asr import IndicConformerNeMoASR
m = IndicConformerNeMoASR('/app/models/indicconformer-nemo/indicconformer_stt_pa_hybrid_rnnt_large.nemo')
print('LOADED OK')
