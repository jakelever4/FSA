from google.cloud import language_v1
from google.cloud.language_v1 import enums
from google.oauth2 import service_account

# set google application credentials
credentials = service_account.Credentials.from_service_account_file("fire-sentiment-analysis-bf24604da498.json")


def analyze_entities(text_content):
    client = language_v1.LanguageServiceClient(credentials=credentials)

    # Available types: PLAIN_TEXT, HTML
    type_ = enums.Document.Type.PLAIN_TEXT

    document = {"content": text_content, "type": type_}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = enums.EncodingType.UTF8

    response = client.analyze_entities(document, encoding_type=encoding_type)

    entities = []

    # Loop through entitites returned from the API
    for entity in response.entities:
        name = entity.name
        type = enums.Entity.Type(entity.type)
        salience = entity.salience
        metadata_dict = {}
        mentions = []

        for metadata_name, metadata_value in entity.metadata.items():
            data = {metadata_name : metadata_value}
            metadata_dict.update(data)

        for mention in entity.mentions:
            entity_mention = Entity.Entity_Mention(mention.text.content, enums.EntityMention.Type(mention.type).name)
            mentions.append(entity_mention)

        entity_obj = Entity.Entity(name, type, salience, metadata_dict, mentions)
        entities.append(entity_obj)

        # print(entity_obj)

    return entities