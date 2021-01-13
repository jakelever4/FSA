from google.cloud.language_v1 import enums

class Entity:
    def __init__(self, name, type, salience, metadata, mentions):
        self.name = name
        self.type = type
        self.type_name = enums.Entity.Type(self.type).name
        self.salience = salience
        self.metadata = metadata
        self.mentions = mentions

    def __str__(self):
        print("Entity:")
        return "Name: {}. Type: {}. Metadata: {}. salience {}.".format(self.name, self.type_name, self.metadata, self.salience)



class Entity_Mention:
    def __init__(self, text, type):
        self.text = text
        self.type = type

    def __str__(self):
        return "Text: {}, Type: {}".format(self.text, self.type)