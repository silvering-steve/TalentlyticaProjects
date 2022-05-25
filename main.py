import os
import re
import fitz # install
import nltk
import glob
import cv2 # install
import easyocr
import spacy.cli
import textseg as ts
from PyPDF2 import PdfFileReader # install
from pdf2image import convert_from_path # install

# nltk.download('punkt') # 1st run
# nltk.download('stopwords') # 1st run
# spacy.cli.download("en_core_web_lg") # 1st run

# SKILLS EXTRACTION
# Add skills database from a file
def add_skills_data(filePath):
    skills = []

    for data in open(filePath, 'r', encoding='UTF-8'):
        skills.append(data.strip())

    return skills

# Get the text from a file
def extract_text(filePath, remove_line=False):
    with fitz.open(filePath) as doc:
        text = ""
        for page in doc:
            text += page.get_text()

        if remove_line:
            text = text = re.sub('\s', " ", text)

    return text

# Extract the skills based on the skill database
def extract_skills(input_text, skills_data):
    stop_word = set(nltk.corpus.stopwords.words('english'))
    word_tokens = nltk.tokenize.word_tokenize(input_text)

    filtered_tokens = [w for w in word_tokens if w not in stop_word]
    filtered_tokens = [w for w in word_tokens if w.isalpha()]
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))

    skills = set()

    for token in filtered_tokens:
        if token in skills_data:
            skills.add(token)

    for ngram in bigrams_trigrams:
        if ngram in skills_data:
            skills.add(ngram)

    return skills

# Extract skills from a single file
def extract_single_skills(filePath, skills):
    text = extract_text(filePath)

    return  extract_skills(text, skills)

# Extract skills from a folder full of pdf
def extract_batch_skill(filePath, skills):
    data = {"File": [], 'Skill': []}

    for file in glob.glob('{}*.pdf'.format(filePath)):
        text = extract_text(file, True)
        data['File'].append(file)
        data['Skill'].append(extract_skills(text, skills))

    return data

# DOCUMENT SEGMENTATION
# Converting from pdf to image for segmentation
def convert_pdf_to_image(filepath,img_path_to_save):
    try:
        fileName = filepath.split("/")[-1].replace(".pdf","")
        pages = convert_from_path(filepath, 350)
        i = 1
        for page in pages:
            image_name = img_path_to_save+fileName+"Page_" + str(i) + ".png"
            page.save(image_name, "JPEG")
            i = i+1
        return {"status":200,"response":"PDF Converted to image sucessfully","fileName":fileName}
    except Exception as e:
        return {"status":400,"response":str(e)}

def text_from_easyocr(img, reader):
    all_text = ""
    result = reader.readtext(img)

    for (bbox, text, prob) in result:
        all_text += text + " "

    return all_text


# Segment and then extract the data from a resume
def segment_extract_data(data,  path_to_write, reader=None, singleFile=True):
    documents = [] # file path nya untuk pdf

    if singleFile:
        documents.append(data)
    else:
        documents = data

    final_name_list=[] # nama file
    final_text_opencv=[] # text dengan segmen
    final_text_easyocr=[] # semua text tanpa segmen
    for i in documents:
        pdf = PdfFileReader(open(i,'rb'))
        fname = i.split('/')[-1]

        images = convert_from_path(i)
        resumes_img=[]
        for j in range(len(images)):
            # Save pages as images in the pdf
            images[j].save(path_to_write+fname.split('.')[0]+'_'+ str(j) +'.jpg', 'JPEG')
            resumes_img.append(path_to_write+fname.split('.')[0]+'_'+ str(j) +'.jpg')
        name_list = fname.split('.')[0]+'_' +'.jpg'
        text_opencv=[]
        text_easyocr=[]
        for i in resumes_img:
            frame=cv2.imread(i)
            os.remove(i)
            img = i.split("/")[2]

            output_img,label,dilate, c_dict,df1, split_img=ts.get_text_seg(frame, img)
            cv2.imwrite(path_to_write+img.split('.')[0]+".png",output_img)
            for i in range(len(split_img)):
                cv2.imwrite(path_to_write+img.split('.')[0]+str(i)+".png", split_img[i])

            text_opencv.append(c_dict)
            text_easyocr+=text_from_easyocr(output_img, reader)
            easyocr_str = ''.join(text_easyocr)

        final_name_list.append(name_list)
        final_text_opencv.append(text_opencv)
        final_text_easyocr.append(easyocr_str)

    return final_text_opencv, final_name_list, final_text_easyocr

# EXPERIENCE EXTRACTION
# Extract exp from a text
def extract_exp(textList, nlp):
    exp = []

    for i in range(len(textList)):
        for j in range(len(textList[i])):
            for _, text in textList[i][j].items():
                text = re.sub(r'[^\w\s]+', "", text)
                text = re.sub(r'[\s]{2,}', " ", text)
                text = re.sub(r'https\w+', "", text)
                doc = nlp(text)
                if doc.cats['experience'] > 0.70:
                    exp.append(text)

    return exp

# Do all the above with just 1 function
def extract_data(filePath, skills, nlp, temp_path, reader=None):
    file_data = {'File': "", 'Skills':"", "Exp":""}

    textList, fileName, fullText = segment_extract_data(filePath, temp_path, reader)
    file_data['File'] = fileName[0]
    file_data['Skills'] = extract_skills((fullText[0]), skills_data=skills)
    file_data['Exp'] = extract_exp(textList, nlp)

    return file_data

def batch_extract_data(filePath, skills, nlp, temp_path):
    file_data = {'File': [], 'Skills': [], "Exp": []}

    for file in os.listdir(filePath):
        data = extract_data('{}/{}'.format(filePath, file), skills, nlp, temp_path)
        file_data['File'].append(data['File'])
        file_data['Skills'].append(data['Skills'])
        file_data['Exp'].append(data['Exp'])

    return file_data

if __name__ == '__main__':
    # Adding skills database
    skills = add_skills_data('list_of_skills.txt')
    skills[0] = '.NET'  # First skills is not UTF-8 so we need to replace it

    # Segmentation need a temp folder for storing image that will be scanned for extraction the text
    temp_path = ('./segmentation/')

    # Load the machine learning model for exp classification
    nlp = spacy.load('model/model_exp')

    # Reader for OCR
    reader = easyocr.Reader(['en'])

    # File location
    filepath = input("File Path : ")

    # Get the filename, skills, exp
    data = extract_data(filepath, skills, nlp, temp_path, reader)

    # Print data
    print(data)

