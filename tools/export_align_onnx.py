r"""Export torchaudio's WAV2VEC2_ASR_BASE_960H to ONNX. Run once, offline.

Needs torch; the shipped app does not. Run it anywhere with a torch install and
copy the two outputs into models\:

    models\wav2vec2-align.onnx    the model server/align.py loads
    models\wav2vec2-align.json    its label set and sample rate

    python tools\export_align_onnx.py [--int8]

The label set matters: torchaudio's is 29 entries with CTC blank at index 0 and
the word separator at 1, upper case. server/align.py lower-cases it to match.
"""

import argparse
import json

import torch
import torchaudio


class _Wrap(torch.nn.Module):
    """Wav2Vec2Model.forward returns (emissions, lengths); ONNX wants one."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, waveform):
        emissions, _ = self.m(waveform)
        return emissions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="wav2vec2-align.onnx")
    ap.add_argument("--int8", action="store_true",
                    help="also write an int8 copy (4x smaller)")
    a = ap.parse_args()

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().eval()
    torch.onnx.export(
        _Wrap(model), torch.zeros(1, bundle.sample_rate * 5), a.out,
        input_names=["waveform"], output_names=["emissions"],
        dynamic_axes={"waveform": {0: "batch", 1: "samples"},
                      "emissions": {0: "batch", 1: "frames"}},
        opset_version=17, do_constant_folding=True,
    )
    json.dump({"labels": list(bundle.get_labels()),
               "sample_rate": bundle.sample_rate},
              open(a.out.replace(".onnx", ".json"), "w"), indent=1)
    print("wrote", a.out)

    if a.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q = a.out.replace(".onnx", "-int8.onnx")
        quantize_dynamic(a.out, q, weight_type=QuantType.QInt8)
        print("wrote", q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
