import os
import argparse
import pandas as pd


def main():
    dirs = os.listdir(args.directory)
    for run_name in dirs:
        pickle_files = [f for f in os.listdir(os.path.join(args.directory, run_name)) if f.endswith(".pkl")]
        for file in pickle_files:
            data = pd.read_pickle(os.path.join(args.directory, run_name, file))
            df = pd.DataFrame(data)
            frames = df.frame_time.unique()
            df.sort_values(by=['face_index', 'frame_time'], ignore_index=True, inplace=True)
            df = df.set_index('face_index')

            df.reset_index(inplace=True)
            # df = df[['frame_time', 'face_index']]
            df = df.drop_duplicates(['frame_time', 'face_index'])
            dfpiv = df[:].pivot(index='frame_time', columns='face_index', values="embedding_fer_vec")
            faces = [i for i in dfpiv.columns if type(i) == int]
            print(f"{run_name} / {file} : {len(faces)} participants")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyse embbeding data frame.')
    parser.add_argument('--directory', default='/e/data/thessis_movies/analysis2' , help='directory og results')
    args = parser.parse_args()
    main()
