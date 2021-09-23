from google.cloud import language_v1
from google.oauth2 import service_account
import Entity

# set google application credentials
credentials = service_account.Credentials.from_service_account_file("fire-sentiment-analysis-bf24604da498.json")


def analyze_entities(text_content):
    client = language_v1.LanguageServiceClient(credentials=credentials)

    # Available types: PLAIN_TEXT, HTML
    type_ = language_v1.Document.Type.PLAIN_TEXT

    document = {"content": text_content, "type_": type_}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_entities(request={'document': document, 'encoding_type': encoding_type})

    entities = []

    # Loop through entitites returned from the API
    for entity in response.entities:
        name = entity.name
        type = language_v1.Entity.Type(entity.type_).name
        salience = entity.salience
        metadata_dict = {}
        mentions = []

        for metadata_name, metadata_value in entity.metadata.items():
            data = {metadata_name : metadata_value}
            metadata_dict.update(data)

        for mention in entity.mentions:
            entity_mention = Entity.Entitiy_Mention(mention.text.content, language_v1.EntityMention.Type(mention.type_).name)
            mentions.append(entity_mention)

        entity_obj = Entity.Entity(name, type, salience, metadata_dict, mentions)
        entities.append(entity_obj)

        print(entity_obj)

    return entities


text = "#BCWildfire Service is responding to the Brenda Creek wildfire (K51924) currently burning south of the #Okanagan Connector (#BCHwy97C), highly visible from the roadside. ~40km from #WestKelowna, this fire is estimated to be 40 ha and is classified as “Out of Control.”"
entities = analyze_entities(text)
print(entities)