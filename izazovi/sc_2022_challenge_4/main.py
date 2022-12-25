from process import extract_information_from_image, Person
import glob
import sys
import os
import csv

# ------------------------------------------------------------------
# Constants
FIELD_NAMES = ['birth_date', 'expiry_date', 'gender', 'name',
               'surname', 'number', 'filename']
# ------------------------------------------------------------------
# Ovaj fajl ne menjati, da bi automatsko ocenjivanje bilo moguce
if len(sys.argv) > 1:
    VALIDATION_DATASET_PATH = sys.argv[1]
else:
    VALIDATION_DATASET_PATH = '.'+os.path.sep+'dataset'+os.path.sep+'validation'+os.path.sep
# -------------------------------------------------------------------

# izvrsiti citanje teksta sa svih fotografija iz validacionog skupa podataka
processed_image_names = []
extracted_person_data = []

for image_path in glob.glob(VALIDATION_DATASET_PATH + "*.jpg"):
    image_directory, image_name = os.path.split(image_path)
    processed_image_names.append(image_name)
    extracted_person_data.append(extract_information_from_image(image_path))


# -----------------------------------------------------------------
# Kreiranje fajla sa rezultatima ekstrakcije za svaku sliku
result_file_contents = ""
# sacuvaj formirane rezultate u csv fajl
with open('result.csv', 'w') as output_file:
    writer = csv.DictWriter(output_file, fieldnames=FIELD_NAMES)
    writer.writeheader()

    for image_index, image_name in enumerate(processed_image_names):
        person: Person = extracted_person_data[image_index]
        writer.writerow({
            'birth_date': person.birth_date,
            'expiry_date': person.expiry_date,
            'gender': person.gender,
            'name': person.name,
            'surname': person.surname,
            'number': person.number,
            'filename': image_name
        })



# ------------------------------------------------------------------
