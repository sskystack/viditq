A few minor modification need to make for the installed `OpenSORA`

- `RF/__init__.py`: add some logic for precompute_text_embeds:
    - add default `precompute_text_embeds=False` as sample() input attribute

```         
# INFO: save the text embeds to avoid save text_encoder
save_d = {}
save_d['model_args'] = model_args
torch.save(save_d, './precomputed_text_embeds.pth')
```