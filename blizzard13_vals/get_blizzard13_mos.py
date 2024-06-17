import pandas as pd

df = pd.read_csv('extracted_data/blizzard2013/2013-EH2-EXT/test_results/final.tsv', sep='\t')

for score_type in df['score_type'].unique():
    if score_type == 'MOS':
        continue
    df_score = df[df['score_type'] == score_type]
    df_score = df_score.groupby(['system'])[['score']].mean()
    df_score = df_score.reset_index()
    df_score['system'] = df_score['system'].str.replace(' ', '_')
    df_score['system'] = df_score['system'].str.replace('FastPitch_WaveGAN', 'fastpitch_wg')
    df_score['system'] = df_score['system'].str.replace('FastPitch_WaveNet', 'fastpitch_wn')
    df_score['system'] = df_score['system'].str.replace('Natural', 'natural16')
    df_score['system'] = df_score['system'].str.replace('Tacotron_WaveGAN', 'tacotron_wg')
    df_score['system'] = df_score['system'].str.replace('Tacotron_WaveNet', 'tacotron_wn')
    score_type = score_type.lower().replace(' ', '_')
    df_score.to_csv(f'b13_{score_type}.csv', index=False)