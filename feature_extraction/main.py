import os
import glob
from tqdm import tqdm
import pandas as pd
from pprint import pprint

from lbp_feature_extraction import get_lbp_features, get_lbp_feature_names
from tamura_feature_extraction import get_tamura_features, get_tamura_feature_names, get_tamura_on
from glrlm_feature_extraction import get_glrlm_features, get_glrlm_feature_names, get_glrlm_on
from yuv_color_moment import get_yuv_color_moment_features, get_yuv_color_moment_feature_names
from rgb_color_moment import get_rgb_color_moment_features, get_rgb_color_moment_feature_names
from lab_color_moment import get_lab_color_moment_features, get_lab_color_moment_feature_names
from glcm_feature_extraction import get_glcm_features, get_glcm_feature_names

# Ensure the output directory exists
output_data = "../features_data/no_deletion"
os.makedirs(output_data, exist_ok=True)

# Define the folder containing segmented images
segmented_folder = "../segmented_images/multiotsu_seperated_all"
image_folders = glob.glob(os.path.join(segmented_folder, '*'))

# Dictionary to manage feature extraction methods and their metadata
features_dictionary = {
    'lbp': {
        'feature_extractor': get_lbp_features,
        'features_name': get_lbp_feature_names(),
        'features': []
    },
    'tamura': {
        'feature_extractor': get_tamura_features,
        'features_name': get_tamura_feature_names(),
        'features': []
    },
    # 'tamura_on': {
    #     'feature_extractor': get_tamura_on,
    #     'features_name': get_tamura_feature_names(),
    #     'features': []
    # },
    'glrlm': {
        'feature_extractor': get_glrlm_features,
        'features_name': get_glrlm_feature_names(),
        'features': []
    },
    # 'glrlm_on': {
    #     'feature_extractor': get_glrlm_on,
    #     'features_name': get_glrlm_feature_names(),
    #     'features': []
    # },
    'yuv_color_moment': {
        'feature_extractor': get_yuv_color_moment_features,
        'features_name': get_yuv_color_moment_feature_names(),
        'features': []
    },
    'lab_color_moment': {
        'feature_extractor': get_lab_color_moment_features,
        'features_name': get_lab_color_moment_feature_names(),
        'features': []
    },
    'rgb_color_moment': {
        'feature_extractor': get_rgb_color_moment_features,
        'features_name': get_rgb_color_moment_feature_names(),
        'features': []
    },
    # 'glcm': {
    #     'feature_extractor': get_glcm_feature_names,
    #     'features_name': get_glcm_feature_names(),
    #     'features': []
    # },
}

# Initialize lists for storing labels and image IDs
labels = []
image_id = []

# Iterate over each folder of segmented images
for image_folder in image_folders:
    images = glob.glob(os.path.join(image_folder, '*'))
    label = os.path.basename(image_folder)
    
    # Process each image in the folder
    for image in tqdm(images, desc=f'Processing {label}', unit='image'):
        labels.append(label)
        image_id.append(os.path.basename(image)[:-4])

        # Extract features using each feature extraction method
        for feature_extraction_method in features_dictionary:
            feature_extractor = features_dictionary[feature_extraction_method]['feature_extractor']
            feature = feature_extractor(image)
            features_dictionary[feature_extraction_method]['features'].append(feature)

# Combine features and labels into dataframes
df_image_id = pd.DataFrame({'Image': image_id})
df_features = {}

# Process each feature type to create a dataframe
for key in features_dictionary:
    df = pd.DataFrame(features_dictionary[key]['features'], columns=features_dictionary[key]['features_name'])
    df = df.loc[:, (df != df.iloc[0]).any()]
    df_features[key] = df

df_labels = pd.DataFrame(labels, columns=['label'])
df_features['label'] = df_labels
pprint(df_features)

df_color_moment_lbp_glrlm_tamura = pd.concat([
    df_image_id,
    df_features["yuv_color_moment"],
    df_features["lbp"],
    df_features["glrlm"],
    df_features["tamura"],
    df_features["label"],
    ], axis=1)

# df_color_moment_lbp_glrlm_tamura_from_lbp = pd.concat([
#     df_image_id,
#     df_features["yuv_color_moment"],
#     df_features["lbp"],
#     df_features["glrlm_on"],
#     df_features["tamura_on"],
#     df_features["label"],
#     ], axis=1)

# df_color_moment_tamura_from_lbp = pd.concat([
#     df_image_id,
#     df_features["yuv_color_moment"],
#     df_features["tamura_on"],
#     df_features["label"],
#     ], axis=1)

df_color_moment_tamura = pd.concat([
    df_image_id,
    df_features["yuv_color_moment"],
    df_features["tamura"],
    df_features["label"],
    ], axis=1)

# df_rgb_color_moment_glcm = pd.concat([
#     df_image_id,
#     df_features["rgb_color_moment"],
#     df_features["glcm"],
#     df_features["label"],
#     ], axis=1)

# df_glcm = pd.concat([
#     df_image_id,
#     df_features["glcm"],
#     df_features["label"],
#     ], axis=1)

df_rgb_color_moment = pd.concat([
    df_image_id,
    df_features["rgb_color_moment"],
    df_features["label"],
    ], axis=1)

df_color_moment = pd.concat([
    df_image_id,
    df_features["yuv_color_moment"],
    df_features["label"],
    ], axis=1)

df_lbp = pd.concat([
    df_image_id,
    df_features["lbp"],
    df_features["label"],
    ], axis=1)

df_glrlm = pd.concat([
    df_image_id,
    df_features["glrlm"],
    df_features["label"],
    ], axis=1)

df_tamura = pd.concat([
    df_image_id,
    df_features["tamura"],
    df_features["label"],
    ], axis=1)

# df_glrlm_from_lbp = pd.concat([
#     df_image_id,
#     df_features["glrlm_on"],
#     df_features["label"],
#     ], axis=1)

# df_tamura_from_lbp = pd.concat([
#     df_image_id,
#     df_features["tamura_on"],
#     df_features["label"],
#     ], axis=1)

df_lab_color_moment_lbp_glrlm_tamura = pd.concat([
    df_image_id,
    df_features["lab_color_moment"],
    df_features["lbp"],
    df_features["glrlm"],
    df_features["tamura"],
    df_features["label"],
    ], axis=1)

# df_lab_moment_lbp_glrlm_tamura_from_lbp = pd.concat([
#     df_image_id,
#     df_features["lab_color_moment"],
#     df_features["lbp"],
#     df_features["glrlm_on"],
#     df_features["tamura_on"],
#     df_features["label"],
#     ], axis=1)

df_lab_color_moment = pd.concat([
    df_image_id,
    df_features["lab_color_moment"],
    df_features["label"],
    ], axis=1)

df_rgb_color_moment_lbp_glrlm_tamura = pd.concat([
    df_image_id,
    df_features["rgb_color_moment"],
    df_features["lbp"],
    df_features["glrlm"],
    df_features["tamura"],
    df_features["label"],
    ], axis=1)

# df_rgb_color_moment_lbp_glrlm_tamura_from_lbp = pd.concat([
#     df_image_id,
#     df_features["rgb_color_moment"],
#     df_features["lbp"],
#     df_features["glrlm_on"],
#     df_features["tamura_on"],
#     df_features["label"],
#     ], axis=1)

# df_rgb_color_moment_tamura_from_lbp = pd.concat([
#     df_image_id,
#     df_features["rgb_color_moment"],
#     df_features["tamura_on"],
#     df_features["label"],
#     ], axis=1)

df_rgb_color_moment_tamura = pd.concat([
    df_image_id,
    df_features["rgb_color_moment"],
    df_features["tamura"],
    df_features["label"],
    ], axis=1)

# df_lab_color_moment_tamura_from_lbp = pd.concat([
#     df_image_id,
#     df_features["lab_color_moment"],
#     df_features["tamura_on"],
#     df_features["label"],
#     ], axis=1)

df_lab_color_moment_tamura = pd.concat([
    df_image_id,
    df_features["lab_color_moment"],
    df_features["tamura"],
    df_features["label"],
    ], axis=1)

df_color_moment_lbp_glrlm_tamura.sort_values(by='Image').to_csv(os.path.join(output_data, 'YUV + LBP + GLRLM + TAMURA.csv'), index=False)
# df_color_moment_lbp_glrlm_tamura_from_lbp.sort_values(by='Image').to_csv(os.path.join(output_data, 'YUV + LBP + GLRLM + TAMURA (From LBP).csv'), index=False)
# df_color_moment_tamura_from_lbp.sort_values(by='Image').to_csv(os.path.join(output_data, 'YUV + TAMURA (From LBP).csv'), index=False)
df_color_moment_tamura.sort_values(by='Image').to_csv(os.path.join(output_data, 'YUV + TAMURA.csv'), index=False)
df_color_moment.sort_values(by='Image').to_csv(os.path.join(output_data, 'YUV.csv'), index=False)

df_lab_color_moment_lbp_glrlm_tamura.sort_values(by='Image').to_csv(os.path.join(output_data, 'LAB + LBP + GLRLM + TAMURA.csv'), index=False)
# df_lab_moment_lbp_glrlm_tamura_from_lbp.sort_values(by='Image').to_csv(os.path.join(output_data, 'LAB + LBP + GLRLM + TAMURA (From LBP).csv'), index=False)
# df_lab_color_moment_tamura_from_lbp.sort_values(by='Image').to_csv(os.path.join(output_data, 'LAB + TAMURA (From LBP).csv'), index=False)
df_lab_color_moment_tamura.sort_values(by='Image').to_csv(os.path.join(output_data, 'LAB + TAMURA.csv'), index=False)
df_lab_color_moment.sort_values(by='Image').to_csv(os.path.join(output_data, 'LAB.csv'), index=False)

df_rgb_color_moment_lbp_glrlm_tamura.sort_values(by='Image').to_csv(os.path.join(output_data, 'RGB + LBP + GLRLM + TAMURA.csv'), index=False)
# df_rgb_color_moment_lbp_glrlm_tamura_from_lbp.sort_values(by='Image').to_csv(os.path.join(output_data, 'RGB + LBP + GLRLM + TAMURA (From LBP).csv'), index=False)
# df_rgb_color_moment_tamura_from_lbp.sort_values(by='Image').to_csv(os.path.join(output_data, 'RGB + TAMURA (From LBP).csv'), index=False)
df_rgb_color_moment_tamura.sort_values(by='Image').to_csv(os.path.join(output_data, 'RGB + TAMURA.csv'), index=False)
# df_rgb_color_moment_glcm.sort_values(by='Image').to_csv(os.path.join(output_data, 'RGB + GLCM.csv'), index=False)
df_rgb_color_moment.sort_values(by='Image').to_csv(os.path.join(output_data, 'RGB.csv'), index=False)

# df_tamura_from_lbp.sort_values(by='Image').to_csv(os.path.join(output_data, 'TAMURA (From LBP).csv'), index=False)
# df_glrlm_from_lbp.sort_values(by='Image').to_csv(os.path.join(output_data, 'GLRLM (From LBP).csv'), index=False)
df_tamura.sort_values(by='Image').to_csv(os.path.join(output_data, 'TAMURA.csv'), index=False)
df_glrlm.sort_values(by='Image').to_csv(os.path.join(output_data, 'GLRLM.csv'), index=False)
df_lbp.sort_values(by='Image').to_csv(os.path.join(output_data, 'LBP.csv'), index=False)
# df_glcm.sort_values(by='Image').to_csv(os.path.join(output_data, 'GLCM.csv'), index=False)

"""
Documentation:

1. `features_dictionary`: A dictionary storing feature extraction methods, feature names, and storage for extracted features.
    - Keys are feature types (e.g., 'lbp', 'tamura').
    - Values are dictionaries with keys `feature_extractor`, `features_name`, and `features`.

2. Image Processing:
    - `image_folders`: Globbed paths to folders containing segmented images.
    - Labels are derived from folder names.
    - Features are extracted for each image using the specified feature extraction methods.

3. Dataframes:
    - `df_features`: Dictionary storing pandas DataFrames for each feature type.
    - `df_color_moment_lbp_glrlm_tamura`: A combined dataset with selected features.

4. CSV Export:
    - Final datasets are saved as CSV files to the specified output directory.
"""
