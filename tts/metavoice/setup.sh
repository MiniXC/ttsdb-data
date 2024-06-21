# Python >=3.10,<3.12
git clone https://github.com/metavoiceio/metavoice-src.git
cd metavoice-src
pip install -e .
# remove lines 51-53 in metavoice-src/fam/llm/fast_inference_utils.py
# if "torch._inductor.config.fx_graph_cache" is in the file
# it's an experimental feature that is not supported in torch 2.1.0
if grep -q "torch._inductor.config.fx_graph_cache" fam/llm/fast_inference_utils.py; then
    sed -i '51,53d' fam/llm/fast_inference_utils.py
fi