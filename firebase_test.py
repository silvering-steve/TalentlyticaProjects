import firebase_admin
from firebase_admin import db

def fire_database(certificate, databaseURL, pathData):
    # Add firebase credential
    cred = firebase_admin.credentials.Certificate(certificate)

    # Create the database app
    app = firebase_admin.initialize_app(cred, {
        'databaseURL' : databaseURL
    })

    # Set the reference path for database
    ref = db.reference(pathData)

    return ref

def set_data(ref, data):
    # Set data to database
    cond = False

    while not cond:
        try:
            ref.set(data)
            print("Upload Successful")
            cond = True
        except:
            print("Upload Failed")
            cond = False

# Database variable
# cred = "firebase/lexical-micron-342010-firebase-adminsdk-ggtjz-2211d8aed1.json"
# database_url = 'https://lexical-micron-342010-default-rtdb.asia-southeast1.firebasedatabase.app/'
# database_path = '/'
# data = {'File': 'Alya Ataya_CV - Alya Ataya_.jpg', 'Skills': ['Power Point', 'Operations', 'PIC', 'Customer Service', 'Scheduling', 'Word', 'Agribusiness', 'Conducting', 'Soft Skills', 'Microsoft Excel', 'Excel', 'Member Development', 'Service Operations', 'Organization'], 'Exp': ['Cariilmu Jakarta Manager of Customer Service Set the customer service staff schedule Host Admin Qontak Agent Report Qontak Assist or replace the role of Customer Service Staff when needed including replying to Qontak Scheduling Includes create zoom links telegram groups Follow Up participants who havent passed the quiz and havent done the quiz Send the data of participants who passed the quiz to the platform Maubelajarapa Cariilmu Jakarta Operation Product Development Intern Host and admin prework webinars Update webinar participants through google spreadsheets Making module and conduct research related to competitors Punya Karya Jakarta Business Development intern Group listing for Micro Small and Medium Enterprises Create customer communication framework Acquisition training and evaluation as well as conducting research and developing results The Shonet Jakarta Creative Intern Assistant stylist in fashion and products in the photoshoot Check in check out the products brand Conducting brand that will take a photoshoot and Become a PIC for the brand in the studio Helped the Merchant Team to bulk and rename photos and quality control ', 'Press Institute Surakarta Staff of Member Organization Development Implementing the project provided such as upgrading new members and looking for journalism news Staff of Member Organization Development Surakarta Student Executive Board Conducting activities such as training of internal members training of new students and study orientation Head of Member Development Surakarta Press Institute Sebelas Maret University Leading 6 members and conduct training for internal members ']}


# # Database object
# ref = fire_database(cred, database_url, database_path)
# ref.set(data)