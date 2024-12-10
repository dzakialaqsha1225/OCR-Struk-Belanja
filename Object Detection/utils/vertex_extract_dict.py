import vertexai
from vertexai.preview.generative_models import GenerativeModel
import json
import re
import os

def extract_dict(receipt_ocr: str):
    """
        Extract relevant informations parsed from an ocr of a receipt into a structured dictionary

    Args:
        receipt_ocr (string): a string from applying OCR to a shopping receipt

    Returns:
        data: a dictionary of relevant informations, the contain is provided below in the prompt:
    """

    prompt = '''Provided below is an OCR of Indonesian shopping receipt. You will extract the relevant informations from the OCR text.

    Please return a string that needs to follow the format below (only return the dictionary string and nothing else).
    The returned string should be in the format of how one would define python dictionaries. Also perform typo correction on the OCR text
    before parsing it.

    Do not include escape characters in the values inside the dictionary. If any part in the OCR text contain single or double quotation, drop them.


    Dictionary format:
    extracted_information = {

    "purchase_date" : [], #(String) In ISO 8601 format, just one value in this key

    "purchase_address" : [], #(String) from Indonesian address format, return both the vendor name and its address in one string, just one value in this key. Do not use escape characters.

    "product_name" : [], #(String) directly from the receipt, you need to pass all the products listed in the receipt here, more than one value

    "purchase_price" : [] #(Float) directly from the receipt, same length as the product_name key

    "product_type" : [] #(String) product type of the product, refer to the product type list below, and predict only using the categories provided

    }



    product_type_reference = ["minuman manis", "minuman sehat", "personal hygiene", "makanan manis", "makanan gurih",

    "unknown", "makanan pokok", "produk dewasa"

    ]



    OCR text:

    '''
    
    prompt = prompt + receipt_ocr

    prompt = prompt.replace("\'", '')

    PROJECT_ID = "capstone-bangkit-d0ca4"
    REGION = "us-central1"
    vertexai.init(project=PROJECT_ID, location=REGION)

    generative_multimodal_model = GenerativeModel("gemini-1.5-pro-002")
    response = generative_multimodal_model.generate_content([prompt])

    text = response.candidates[0].content.parts
    text = text[0].text

    json_string = re.search(r'\{.*\}', text, re.DOTALL).group(0)
    json_string = json_string.replace("'", '"')

    with open('llm_output.json', 'w') as file: #need this to capture output, do not remove
        file.write(json_string)

    with open('llm_output.json', 'r') as file:
        data = json.load(file)

    os.remove('llm_output.json')

    return data
