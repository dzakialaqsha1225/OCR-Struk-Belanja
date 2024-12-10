import warnings
import pandas as pd
import re
import object_localization as ol
import vertex_extract_dict as ved
import product_recommender as pr
import cheap_close as cc

def full_deployment(key_path: str, test_path: str, dataset_path: str, uid: str, email: str, model, lon: float, lat: float):
  """
    Takes a picture of a receipt, perform object localization for the receipt, use ocr on cropped localized image,
    then generate recommended places to get similar items for cheaper and closer.

    To be used only after importing all the other utility scripts

    Args:
      key_path (str): Path to the Google Cloud service account JSON key file.
      test_path (str): Path to the image file of the receipt.
      dataset_path(str): Path to the purchase history dataset.
      uid (str): User ID.
      email (str): User email address.
      model (Any): The object detection model to be used for receipt localization.
                  The specific type depends on your implementation.
      lon (float): User's longitude coordinate.
      lat (float): User's latitude coordinate.
    Return:
      a pd dataframe sorted by distance from user's location, offering the cheapest price, at the most up-to-date
      of user's previously purchased items and recommended items based on RFM analysis
  """
  df = pd.read_csv(dataset_path)
  
  if df.empty:
    raise ValueError("DataFrame is empty. Please check the dataset file in the provided path.")
  if 'uid' not in df.columns:
    raise ValueError("uid column is missing from the dataset")
  if re.fullmatch(r"^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$", email) is None:
    raise ValueError('Email is not valid')
  if 'lon' not in df.columns:
    raise ValueError("long column is missing from the dataset")
  if 'lat' not in df.columns:
    raise ValueError("lat column is missing from the dataset")
  
  warnings.simplefilter(action='ignore', category=FutureWarning)
  struk = ol.ocr_receipt(test_path, model) #uses util
  data = ved.extract_dict(struk, key_path, uid, email) #uses util
  data = pd.DataFrame(data)

  df = pd.concat([df, data], ignore_index=True)
  df.to_csv('/content/OCR-Struk-Belanja/recommender/dataset/purchase_history.csv', index=False)
  test_rec = pr.recommend("/content/OCR-Struk-Belanja/recommender/dataset/purchase_history.csv", uid) #uses util
  end_rec = cc.cheap_proximity_rec(
    dataset = "/content/OCR-Struk-Belanja/recommender/dataset/purchase_history.csv",
    uid = test_uid,
    product_list = test_rec, lon = lon, lat = lat)
  return end_rec
