import pandas as pd


csv_file_path = "train.csv"
df = pd.read_csv(csv_file_path)


augmented_data = []
for index, row in df.iterrows():
    sample_name = row['filename']
    class_name = row['label']

    augmented_sample_name = sample_name.split('.')[0] + "_auged.jpg"

    augmented_data.append([sample_name, class_name])
    augmented_data.append([augmented_sample_name, class_name])


augmented_df = pd.DataFrame(augmented_data, columns=['filename', 'label'])
output_csv_file_path = "train_auged.csv"
augmented_df.to_csv(output_csv_file_path, index=False)

print("new train.csv saved:", output_csv_file_path)
