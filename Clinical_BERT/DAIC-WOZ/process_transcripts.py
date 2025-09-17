# 1. merge transcript
import os
import pandas as pd

def process_all_transcripts(data_dir='data'):
    rows = []

    for folder in os.listdir(data_dir):
        session_path = os.path.join(data_dir, folder)
        if not os.path.isdir(session_path) or not folder.endswith("_P"):
            continue

        session_id = folder.replace('_P', '')
        transcript_file = os.path.join(session_path, f'{session_id}_TRANSCRIPT.csv')

        if not os.path.exists(transcript_file):
            print(f"Transcript not found for session {session_id}")
            continue

        # Try reading with tab delimiter first
        try:
            df = pd.read_csv(transcript_file, delimiter='\t')
        except Exception as e:
            print(f"Failed to read {transcript_file} with tab delimiter: {e}")
            continue

        print(f"\nColumns in {session_id}_TRANSCRIPT.csv:", list(df.columns))

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        if 'speaker' not in df.columns or 'value' not in df.columns:
            print(f"⚠️ Skipping session {session_id}: Required columns missing.")
            continue

        # Filter to only participant speech
        df = df[df['speaker'] == 'Participant']

        for _, row in df.iterrows():
            rows.append({
                'session_id': session_id,
                'start_time': row['start_time'],
                'stop_time': row['stop_time'],
                'text': row['value']
            })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = process_all_transcripts('data')
    print(f"\n Processed {len(df)} utterances.")
    df.to_csv('merged_transcripts.csv', index=False)
