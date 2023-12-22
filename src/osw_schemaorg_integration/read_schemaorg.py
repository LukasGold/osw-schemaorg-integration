import copy
import itertools
import json
import re
import urllib.request
import uuid as uuid_lib
from contextlib import contextmanager
from pathlib import Path

import deepl
import osw.data.import_utility as diu
import osw.wiki_tools as wt
import pandas as pd
from osw.model.static import OswBaseModel
from typing_extensions import Dict, List, Optional, Union

# Definition of constants
CREDENTIALS_FILE_PATH_DEFAULT = Path(
    r"C:\Users\gold\ownCloud\Personal\accounts.pwd.yaml"
)
SCHEMAORG_BASE_URL = "https://schema.org/"
SCHEMAORG_VERSION = "latest"
SCHEMAORG_NAMESPACE = "schema"
SCHEMAORG_UUID_NAMESPACE = uuid_lib.UUID("33af577c-44a3-4c41-91a9-942cc08195ba")

# schema.org DataType to JSON Schema type mapping
DATATYPE_MAPPING = {
    "schema:Boolean": {
        "type": "boolean",
        "format": "checkbox",
        "at_type": "xsd:boolean",
    },
    "schema:CssSelectorType": {"type": "string", "at_type": "xsd:string"},
    "schema:Date": {"type": "string", "format": "date", "at_type": "xsd:date"},
    "schema:DateTime": {
        "type": "string",
        "format": "date-time",
        "at_type": "xsd:dateTime",
    },
    "schema:Float": {"type": "number", "at_type": "xsd:float"},
    "schema:Integer": {"type": "integer", "at_type": "xsd:integer"},
    "schema:Number": {"type": "number", "at_type": "xsd:float"},
    # todo: introduce PronounceableText to OSW later on
    "schema:PronounceableText": {"type": "string", "at_type": "xsd:string"},
    "schema:Text": {"type": "string", "at_type": "xsd:string"},
    "schema:Time": {"type": "string", "format": "time", "at_type": "xsd:time"},
    "schema:URL": {"type": "string", "format": "uri", "at_type": "xsd:anyURI"},
    "schema:XPathType": {"type": "string", "at_type": "xsd:string"},
    "other": {"type": "string", "format": "autocomplete", "at_type": "@id"},
}

AUTO_COMPLETE_TEMPLATE = {
    "single_cat_table": {
        "type": "array",
        "format": "table",
        "items": {
            "type": "string",
            "format": "autocomplete",
            "range": "Category:OSW_Category",
            "uniqueItems": True,
            "default": "",
        },
    },
    "multi_cat_table": {
        "type": "array",
        "format": "table",
        "items": {
            "type": "string",
            "format": "autocomplete",
            "options": {
                "autocomplete": {
                    "query": "[[Category:OSW_Category]] OR "
                    "[[Category:OSW_Category_other]] "
                    "|?Display_title_of=label"
                }
            },
            "uniqueItems": True,
            "default": "",
        },
    },
    "single_cat_enty": {
        "type": "string",
        "format": "autocomplete",
        "range": "Category:OSW_Category",
        "default": "",
    },
    "multi_cat_entry": {
        "type": "string",
        "format": "autocomplete",
        "options": {
            "autocomplete": {
                "query": "[[Category:OSW_Category]] OR "
                "[[Category:OSW_Category_other]] "
                "|?Display_title_of=label"
            }
        },
        "default": "",
    },
}


# Definition of classes
class Category(OswBaseModel):
    class_dict: dict
    schemaorg_jsondata: Optional[dict]
    jsondata_slot: Optional[dict] = None
    jsonschema_slot: Optional[dict] = None
    uuid: Optional[uuid_lib.UUID] = None
    osw_id: Optional[str] = None
    subclass_of: Optional[list] = None

    def model_post_init(self):
        self.schemaorg_jsondata = self.class_dict["schemaorg"]
        self.uuid = gen_uuid(self.schemaorg_jsondata["rdfs:label"])
        self.osw_id = diu.uuid_to_full_page_title(uuid=self.uuid, wiki_ns="Category")
        self.subclass_of = [
            id_to_full_page_title(id_=id__)
            for id__ in dict_or_list_to_list(
                dict_or_list=self.schemaorg_jsondata["rdfs:subClassOf"], val_key="@id"
            )
        ]

    def init_jsondata_slot(self):
        split_label = split_pascal_case_string(self.schemaorg_jsondata["rdfs:label"])
        label = " ".join(split_label)
        self.jsondata_slot = {
            "type": ["Category:Category"],
            "subclass_of": self.subclass_of,
            "uuid": str(self.uuid),
            "label": [
                {"text": label, "lang": "en"},
                {
                    "text": translate_with_deepl(
                        label, trans_res=translation_resources
                    ),
                    "lang": "de",
                },
            ],
            "name": self.schemaorg_jsondata["rdfs:label"],
            "description": [
                {"text": self.schemaorg_jsondata["rdfs:comment"], "lang": "en"},
                {
                    "text": translate_with_deepl(
                        self.schemaorg_jsondata["rdfs:comment"],
                        trans_res=translation_resources,
                    ),
                    "lang": "de",
                },
            ],
            "instance_rdf_type": [self.schemaorg_jsondata["@id"]],
        }

    def init_jsonschema_slot(self, properties_dict: dict):
        self.jsonschema_slot = {
            "@context": [{}]
            + [f"/wiki/{sco}?action=raw&slot=jsonschema" for sco in self.subclass_of],
            "allOf": [
                {"$ref": f"/wiki/{sco}?action=raw&slot=jsonschema"}
                for sco in self.subclass_of
            ],
            "type": "object",
            "uuid": str(self.uuid),
            "title": self.schemaorg_jsondata["rdfs:label"],
            "title*": {
                "en": self.schemaorg_jsondata["rdfs:label"],
                "de": translate_with_deepl(
                    self.schemaorg_jsondata["rdfs:label"],
                    trans_res=translation_resources,
                ),
            },
            "description": self.schemaorg_jsondata["rdfs:comment"],
            "description*": {
                "en": self.schemaorg_jsondata["rdfs:comment"],
                "de": translate_with_deepl(
                    self.schemaorg_jsondata["rdfs:comment"],
                    trans_res=translation_resources,
                ),
            },
            "defaultProperties": [],
            "required": ["type"],
            "properties": {
                "type": {"default": [self.osw_id]},
            },
        }

        # todo: Append property mapping to jsonschema_slot @context

    def post_init(self):
        self.model_post_init()
        self.init_jsondata_slot()
        self.init_jsonschema_slot()


# Definition of functions
@contextmanager
def managed_json_file(json_fp: Union[str, Path]):
    with open(json_fp, "r", encoding="utf-8") as f:
        json_data_ = json.load(
            f,
        )
    try:
        yield json_data_
    finally:
        with open(json_fp, "w", encoding="utf-8") as f:
            json.dump(json_data_, f, ensure_ascii=False, indent=4)


def download_json(url):
    """
    Downloads JSON data from the specified URL.
    """
    with urllib.request.urlopen(url) as response:
        data = json.load(response)
    return data


def get_type(entry):
    """
    Key function to extract the "@type" property from an entry.
    Handles cases where "@type" is a list.
    """
    types = entry.get("@type", [])
    return types[0] if isinstance(types, list) else types


def gen_uuid(
    label_: str, namespace: uuid_lib.UUID = SCHEMAORG_UUID_NAMESPACE
) -> uuid_lib.UUID:
    """

    Parameters
    ----------
    label_:
        A label to generate a UUID from. Exp: "Action" in "schema:Action"
    namespace:
        An UUID to serve as namespace for uuid5() generation

    Returns
    -------

    """
    return uuid_lib.uuid5(namespace, label_)


def id_to_full_page_title(id_: str, wiki_ns: str = "Category"):
    label = id_.split(":")[1]
    uuid = gen_uuid(label)
    return diu.uuid_to_full_page_title(uuid=uuid, wiki_ns=wiki_ns)


def dict_or_list_to_list(dict_or_list: Union[dict, list], val_key: str = "@id"):
    if isinstance(dict_or_list, dict):
        return [dict_or_list[val_key]]
    elif isinstance(dict_or_list, list):
        return [element[val_key] for element in dict_or_list]


def get_deepl_translator(
    credentials_fp: Union[str, Path] = None, prev_trans: dict = None
) -> Dict[str, Union[deepl.Translator, dict]]:
    if credentials_fp is None:
        credentials_fp = CREDENTIALS_FILE_PATH_DEFAULT
    if prev_trans is None:
        prev_trans = {}
    domains, accounts = wt.read_domains_from_credentials_file(credentials_fp)
    domain = "api-free.deepl.com"
    auth = accounts[domain]["password"]
    translator = deepl.Translator(auth)
    return {"translator": translator, "prev_trans": prev_trans}


def translate_with_deepl(
    to_translate: str,
    target_lang: str = "DE",
    trans_res: Dict[str, Union[deepl.Translator, dict]] = None,
    credentials_fp: Union[str, Path] = None,
) -> str:
    if credentials_fp is None:
        credentials_fp = CREDENTIALS_FILE_PATH_DEFAULT
    if trans_res is None:
        trans_res = get_deepl_translator(credentials_fp=credentials_fp)
    translator = trans_res["translator"]
    prev_tans = trans_res["prev_trans"]
    if to_translate in prev_tans.keys():
        return prev_tans.get(to_translate)
    else:
        translation = translator.translate_text(
            to_translate, target_lang=target_lang
        ).text
        prev_tans[to_translate] = translation
        return translation


def split_camel_case_string(s: str) -> List[str]:
    return re.findall(r"[A-Z]?[a-z]+", s)


def split_pascal_case_string(s: str) -> List[str]:
    result = re.findall(r"[A-Z]+[a-z]+", s)
    if len(result) == 0:
        # apparently the string is not in PascalCase
        result = [s]
    return result


def get_value_flex(
    str_list_dict: Union[str, list, dict], val_key: str = "@value", lang: str = "en"
) -> str:
    if isinstance(str_list_dict, str):
        return str_list_dict
    elif isinstance(str_list_dict, list):
        for ele in str_list_dict:
            if ele["@language"] == lang:
                return ele[val_key]
    elif isinstance(str_list_dict, dict):
        return str_list_dict[val_key]


def transform_property(prop: dict):
    """Transforms a schema.org property into a property suited for the jsonschema_slot.
    This might also lead to two properties, e.g. if the schema.org property is of mixed
    type (class and datatype).

    Parameters
    ----------
    prop

    Returns
    -------

    """
    id_ = prop["@id"]
    range_includes: Union[dict, list, None] = prop.get("schema:rangeIncludes", None)
    if range_includes is None:
        return None
    label = get_value_flex(str_list_dict=prop["rdfs:label"])
    prop_label = "_".join([s.lower() for s in split_camel_case_string(label)])
    semantic_prop_name = "Has" + "".join(
        [s.capitalize() for s in split_camel_case_string(label)]
    )
    prop_title = " ".join([s.capitalize() for s in split_camel_case_string(label)])
    prop_title_de = translate_with_deepl(
        to_translate=prop_title, trans_res=translation_resources
    )
    prop_desc = get_value_flex(str_list_dict=prop["rdfs:comment"])
    prop_desc_de = translate_with_deepl(
        to_translate=prop_desc, trans_res=translation_resources
    )
    # List of range_includes entries
    ri_id_list = dict_or_list_to_list(dict_or_list=range_includes, val_key="@id")
    # Number of range_includes entries that are datatypes. For each entry
    #   a separate property will be created, because we want to avoid oneOf
    #   properties in the jsonschema            #
    ri_cnt = len(ri_id_list)
    ri_dt_list = [
        ri_entry for ri_entry in ri_id_list if ri_entry in list(DATATYPE_MAPPING.keys())
    ]
    ri_dt_cnt = len(ri_dt_list)
    ri_class_list = [
        ri_entry
        for ri_entry in ri_id_list
        if ri_entry not in list(DATATYPE_MAPPING.keys())
    ]
    # ri_class_cnt = len(ri_class_list)

    # todo:
    #  - create one property for each range_includes entry that is a datatype
    #  - create one property for all range_includes entries that are a class

    def process_ri_dt(dt_str: str, single_prop: bool = False):
        """Creates a json object property and context entries for a schema.org property
        that has one or more rangeIncludes entries that are subclasses of DataTypes.

        Parameters
        ----------
        dt_str:
            The DataType @id string. Should be one of the keys of DATATYPE_MAPPING
        single_prop:
            Whether the DataType is the only rangeIncludes entry

        Returns
        -------

        """
        range_type = dt_str.split(":")[1]
        range_type_lower = "_".join(
            [s.lower() for s in split_pascal_case_string(range_type)]
        )
        if single_prop:
            new_prop_label = prop_label
            new_prop_title = prop_title
            new_prop_title_de = prop_title_de
        else:
            new_prop_label = f"{prop_label}_{range_type_lower}"
            new_prop_title = f"{prop_title} ({range_type})"
            new_prop_title_de = f"{prop_title_de} ({range_type})"
        obj_prop = {
            "title": new_prop_title,
            "title*": {"de": new_prop_title_de},
            "description": prop_desc,
            "description*": {"de": prop_desc_de},
            "type": DATATYPE_MAPPING[dt_str]["type"],
        }
        # todo: discuss whether to use type directly or nested inside a list with items?
        if "format" in DATATYPE_MAPPING[dt_str].keys():
            obj_prop["format"] = DATATYPE_MAPPING[dt_str]["format"]
        # Create context entry
        context_entries_dt = [
            # Create context entry for the schema.org property
            {"@id": id_, "@type": DATATYPE_MAPPING[dt_str]["at_type"]},
            {
                "@id": f"Property:{semantic_prop_name}",
                "@type": DATATYPE_MAPPING[dt_str]["at_type"],
            },
        ]
        return {
            "object_property_name": new_prop_label,
            "object_property": obj_prop,
            "context_entries": context_entries_dt,
        }

    def process_ri_class(class_strs: List[str]):
        """Creates a json object property and context entries for a schema.org property
        that has one or more rangeIncludes entries that are classes (not DataType).

        Parameters
        ----------
        class_strs:
            The list of class @id strings. Should not be one of the keys of
            DATATYPE_MAPPING

        Returns
        -------

        """
        obj_prop = {
            "title": prop_title,
            "title*": {"de": prop_title_de},
            "description": prop_desc,
            "description*": {"de": prop_desc_de},
        }
        # Set type and format (from AUTO_COMPLETE_TEMPLATE) and replace range
        #  / options.autocomplete.query
        if len(class_strs) == 1:
            template_dict = copy.deepcopy(AUTO_COMPLETE_TEMPLATE["single_cat_table"])
            template_dict["items"]["range"] = id_to_full_page_title(
                id_=class_strs[0], wiki_ns="Category"
            )
            obj_prop = {**obj_prop, **template_dict}
        else:
            template_dict = copy.deepcopy(AUTO_COMPLETE_TEMPLATE["multi_cat_table"])
            template_dict["items"]["options"]["autocomplete"]["query"] = (
                " OR ".join(
                    [
                        f"[[{id_to_full_page_title(id_=ri_id_str, wiki_ns='Category')}]]"
                        for ri_id_str in class_strs
                    ]
                )
                + " |?Display_title_of=label"
            )
            obj_prop = {**obj_prop, **template_dict}
        # Create context entries
        context_entries_class = [
            # Create context entry for the schema.org property
            {"@id": id_, "@type": DATATYPE_MAPPING["other"]["at_type"]},
            # Create semantic context entry
            {"@id": f"Property:{semantic_prop_name}", "@type": "@id"},
        ]
        return {
            "object_property_name": prop_label,
            "object_property": obj_prop,
            "context_entries": context_entries_class,
        }

    returned = []
    # List object property entries and context entries
    object_property_entries = {}
    context_entries_ = {}
    # Process rangeIncludes entries
    # All (one!) rangeIncludes entries are datatypes
    if ri_dt_cnt == ri_cnt and ri_dt_cnt == 1:
        returned.append(process_ri_dt(dt_str=ri_dt_list[0], single_prop=True))
    # There are multiple rangeIncludes entries, not necessary all datatypes
    else:
        # Process rangeIncludes entries that are datatypes
        for ri_entry in ri_dt_list:
            returned.append(process_ri_dt(dt_str=ri_entry, single_prop=False))
        # Process rangeIncludes entries that are classes
        returned.append(process_ri_class(class_strs=ri_class_list))
    # Process the returns
    for ret in returned:
        object_property_entries[ret["object_property_name"]] = ret["object_property"]
        context_property_name = ret["object_property_name"]
        for context_entry_ in ret["context_entries"]:
            # Make sure no property mapping is overwritten
            while context_property_name in context_entries_.keys():
                context_property_name += "*"
            context_entries_[context_property_name] = context_entry_

    return {
        "object_property_entries": object_property_entries,
        "context_entries": context_entries_,
    }


# if "schema:rangeIncludes" is of mixed Type, e.g. a class and a datatype
# we need to have a property named new_prop_name and a property named new_prop

# todo: object property table / items?


def transform_class(class_dict: dict, properties_dict: dict) -> dict:
    """Transforms a schema.org class into a class suited for the jsonschema_slot.

    Parameters
    ----------
    class_dict
    properties_dict

    Returns
    -------

    """
    schemaorg_jsondata = class_dict["schemaorg"]
    label = get_value_flex(schemaorg_jsondata["rdfs:label"])
    split_label = split_pascal_case_string(label)
    human_readable_label = " ".join(split_label)
    human_readable_label_de = (
        translate_with_deepl(human_readable_label, trans_res=translation_resources),
    )
    description = get_value_flex(schemaorg_jsondata["rdfs:comment"])
    description_de = translate_with_deepl(description, trans_res=translation_resources)
    uuid = gen_uuid(label)
    osw_id = diu.uuid_to_full_page_title(uuid=uuid, wiki_ns="Category")
    subclass_of_id = []
    if "rdfs:subClassOf" in schemaorg_jsondata.keys():
        subclass_of_id = dict_or_list_to_list(
            dict_or_list=schemaorg_jsondata["rdfs:subClassOf"], val_key="@id"
        )
    subclass_of_labeled = [id_.split(":")[1] for id_ in subclass_of_id]
    subclass_of = [id_to_full_page_title(id_=id__) for id__ in subclass_of_id]

    def create_jsondata_slot():
        return {
            "type": ["Category:Category"],
            "subclass_of": subclass_of,
            "uuid": str(uuid),
            "label": [
                {"text": human_readable_label, "lang": "en"},
                {"text": human_readable_label_de, "lang": "de"},
            ],
            "name": label,
            "description": [
                {"text": description, "lang": "en"},
                {"text": description_de, "lang": "de"},
            ],
            "instance_rdf_type": [schemaorg_jsondata["@id"]],
        }
        # todo: set meta properties

    def create_jsonschema_slot():
        jsonschema_slot = {
            "@context": [{}]
            + [f"/wiki/{sco}?action=raw&slot=jsonschema" for sco in subclass_of],
            "allOf": [
                {"$ref": f"/wiki/{sco}?action=raw&slot=jsonschema"}
                for sco in subclass_of
            ],
            "type": "object",
            "uuid": str(uuid),
            "title": human_readable_label,
            "title*": {"en": human_readable_label, "de": human_readable_label_de},
            "description": description,
            "description*": {"en": description, "de": description_de},
            "defaultProperties": [],
            "required": ["type"],
            "properties": {
                "type": {"default": [osw_id]},
            },
        }
        for prop in class_dict["property"]:
            prop_id = prop["@id"]
            if properties_dict[prop_id]["transformed"] is None:
                continue
            jsonschema_slot["properties"] = {
                **jsonschema_slot["properties"],
                **properties_dict[prop_id]["transformed"]["object_property_entries"],
            }
            jsonschema_slot["@context"][0] = {
                **jsonschema_slot["@context"][0],
                **properties_dict[prop_id]["transformed"]["context_entries"],
            }
        return jsonschema_slot

    return {
        "jsondata_slot": create_jsondata_slot(),
        "jsonschema_slot": create_jsonschema_slot(),
        "uuid": uuid,
        "osw_id": osw_id,
        "subclass_of_labeled": subclass_of_labeled,
        "label": label,
    }


def transform_instances(instance: dict):
    id_str = instance["@id"]
    """The schema.org id, e.g. schema:ActiveActionStatus"""
    type_ = instance["@type"]
    """The schema.org type, e.g. schema:ActionStatusType"""
    type_ref = (
        id_to_full_page_title(type_)
        if isinstance(type_, str)
        else [id_to_full_page_title(tt) for tt in type_]
    )
    """The OSW category, e.g. 'Category:OSW52c2c5a6bbc84fcb8eab0fa69857e7dc"""
    type_ref_uuid = (
        gen_uuid(type_.split(":")[1])
        if isinstance(type_, str)
        else [gen_uuid(tt.split(":")[1]) for tt in type_]
    )
    label = get_value_flex(instance["rdfs:label"])
    uuid = (
        gen_uuid(label, type_ref_uuid)
        if isinstance(type_ref_uuid, uuid_lib.UUID)
        else gen_uuid(label, type_ref_uuid[0])
    )
    split_label = split_pascal_case_string(label)
    human_readable_label = " ".join(split_label)
    human_readable_label_de = translate_with_deepl(
        human_readable_label, trans_res=translation_resources
    )
    description = get_value_flex(instance["rdfs:comment"])
    description_de = translate_with_deepl(description, trans_res=translation_resources)
    osw_id = diu.uuid_to_full_page_title(uuid=uuid, wiki_ns="Item")

    jsondata = {
        "@id": id_str,
        "@type": type_,
        "rdf_type": type_ if isinstance(type_, list) else [type_],
        "type": [type_ref],
        "uuid": str(uuid),
        "label": [
            {"value": human_readable_label, "lang": "en"},
            {"value": human_readable_label_de, "lang": "de"},
        ],
        "name": label,
        "description": [
            {"text": description, "lang": "en"},
            {"text": description_de, "lang": "de"},
        ],
    }

    return {
        "schemaorg": instance,
        "transformed": {
            "jsondata_slot": jsondata,
            "uuid": str(uuid),
            "osw_id": osw_id,
            "instance_of": type_ref,
        },
        "label": label,
        "schemaorg_id": id_str,
        "rdf_type": type_,
    }


def main():
    pass


if __name__ == "__main__":
    with managed_json_file("data/previous_translations.json") as previous_translations:
        # Deepl translator for reuse
        translation_resources = get_deepl_translator(prev_trans=previous_translations)

        # URL for the JSON data
        json_url = "https://schema.org/version/latest/schemaorg-current-https.jsonld"

        # Download the JSON data
        schemaorg_json_data = download_json(json_url)

        # Extract the "@graph" entries
        graph_entries = schemaorg_json_data.get("@graph", [])

        # Sort the entries by "@type"
        sorted_entries = sorted(graph_entries, key=get_type)

        # Sort sorted_entries into sublist according to "@type"
        grouped_entries = {}
        for key, group in itertools.groupby(sorted_entries, key=get_type):
            if key not in grouped_entries.keys():
                grouped_entries[key] = []
                # print("New group:", key)
            for thing in group:
                grouped_entries[key].append(thing)

        # Making the dictionary jsonpath compatible
        grouped_entries_as_dict = {}
        for key, list_ in grouped_entries.items():
            if key not in grouped_entries_as_dict.keys():
                grouped_entries_as_dict[key] = {}
                # print("New group:", key)
            for ele in list_:
                id_ = ele["@id"]
                grouped_entries_as_dict[key][id_] = {"jsondata": ele}

        # Processing of schema.org properties
        properties_without_range_includes = []
        properties = {}
        for property_dict in grouped_entries["rdf:Property"]:
            key = property_dict["@id"]
            transformed_property = transform_property(prop=property_dict)
            if transformed_property is None:
                properties_without_range_includes.append(property_dict)
                continue
            else:
                properties[key] = {
                    "schemaorg": property_dict,
                    "transformed": transformed_property,
                }

        print(f"Number of properties: {len(grouped_entries_as_dict['rdf:Property'])}")
        print(
            f"Number of properties without rangeIncludes attribute "
            f"(potentially deprecated):  {len(properties_without_range_includes)}"
        )

        # Classes to operate on
        classes = {}
        for class_entry in grouped_entries["rdfs:Class"]:
            class_id = class_entry["@id"]
            if class_id not in classes.keys():
                classes[class_id] = {
                    "schemaorg": {},
                    "transformed": {},
                    "subclass": [],
                    "property": [],
                }
            classes[class_id]["schemaorg"] = class_entry
            # Listing subclasses
            subclasses = diu.jsonpath_search_and_return_list_simple(
                jp_str=f'*[?"rdfs:subClassOf"[*]."@id" == "{class_id}"]',
                val_key="@id",
                search_tar=grouped_entries_as_dict["rdfs:Class"],
            )
            classes[class_id]["subclass"] = subclasses
            # Listing properties
            class_properties = diu.jsonpath_search_and_return_list_simple(
                jp_str=f'*[?"schema:domainIncludes"[*]."@id" == "{id_}"]',
                val_key=None,
                search_tar=grouped_entries_as_dict["rdf:Property"],
            )
            classes[class_id]["property"] = class_properties
            # Transforming classes
            classes[class_id]["transformed"] = transform_class(
                class_dict=classes[class_id], properties_dict=properties
            )

        # Listing classes
        schemaorg_classes_list = classes.keys()

        # todo: list classes in a table

        # Processing instances
        instances = {}
        schemaorg_instances = []
        for key, group in grouped_entries.items():
            if key in ["rdf:Property", "rdfs:Class", "schema:DataType"]:
                continue
            schemaorg_instances.extend(group)
        print(f"Number of instances: {len(schemaorg_instances)}")
        for inst in schemaorg_instances:
            retval = transform_instances(inst)
            instances[retval["schemaorg_id"]] = retval

        # todo: list instances in a table

        # A table for checking the properties
        schemaorg_properties_table = pd.DataFrame(
            data={
                "name": list(properties.keys()),
                "@id": [prop["schemaorg"]["@id"] for prop in properties.values()],
                "@type": [prop["schemaorg"]["@type"] for prop in properties.values()],
                "comment": [
                    prop["schemaorg"]["rdfs:comment"] for prop in properties.values()
                ],
                "label": [
                    prop["schemaorg"]["rdfs:label"] for prop in properties.values()
                ],
                "domainIncludes": [
                    dict_or_list_to_list(prop["schemaorg"]["schema:domainIncludes"])
                    for prop in properties.values()
                ],
                "rangeIncludes": [
                    dict_or_list_to_list(prop["schemaorg"]["schema:rangeIncludes"])
                    for prop in properties.values()
                ],
                "transformed": [
                    list(prop["transformed"]["object_property_entries"].keys())
                    for prop in properties.values()
                ]  # todo: why are some transformed properties listed twice?
                #  (once with tailing _ [appearse with type: URL] and once without)
            }
        ).sort_values(by=["name"])
        schemaorg_properties_table.to_excel("data/properties.xlsx")
        # Listing the derived object properties (for displaying in a table)
        object_properties = {}
        # jsonpath compatible dictionary
        object_properties_as_dict = {}
        for prop_key, prop in properties.items():
            for trans_key, trans_prop in prop["transformed"][
                "object_property_entries"
            ].items():
                object_properties[trans_key] = {
                    "name": trans_key,
                    "derived from": prop_key,
                    "title": trans_prop["title"],
                    "title_de": trans_prop["title*"]["de"],
                    "description": trans_prop["description"],
                    "description_de": trans_prop["description*"]["de"],
                    "type": trans_prop["type"],
                    "format": trans_prop.get("format", None),
                }
                object_properties_as_dict[trans_key] = {
                    "jsondata": object_properties[trans_key]
                }

        print(f"Number of object properties: {len(object_properties)}")
        # A table for checking the derived object properties
        object_properties_table = pd.DataFrame(
            data=list(object_properties.values())
        ).sort_values(by=["name"])
        object_properties_table.to_excel("data/object_properties.xlsx")

        # Listing the derived semantic properties
        semantic_properties: Dict[str, Dict[str, str]] = {}
        for prop_key, prop in properties.items():
            if prop["transformed"] is None:
                continue
            context_entries = prop["transformed"]["context_entries"]
            added = False
            used_for = []
            for obj_prop_name, mapping in context_entries.items():
                # Single top level key, context_entries:
                # [{"name*": {"@id": "Property:HasName", "@type": "@id"}}, ...]
                property_name = mapping["@id"]
                if "Property:" not in property_name:
                    continue
                semantic_property_name = property_name.split(":")[1]
                semantic_property_type = mapping["@type"]
                used_for.append(obj_prop_name)

                if semantic_property_name not in semantic_properties.keys():
                    semantic_properties[semantic_property_name] = {
                        "name": semantic_property_name,
                        "type": semantic_property_type,
                        "derived from": prop_key,
                        "used for": used_for,
                        "context entry": mapping,
                    }
                else:
                    if (
                        obj_prop_name
                        not in semantic_properties[semantic_property_name]["used for"]
                    ):
                        semantic_properties[semantic_property_name]["used for"].append(
                            obj_prop_name
                        )
                added = True
            if not added:
                print(f"Property {property_name} not added")

        print(f"Number of semantic properties: {len(semantic_properties)}")
        # A table for checking the derived semantic properties
        semantic_properties_table = pd.DataFrame(
            data=list(semantic_properties.values()),
        ).sort_values(by=["name"])
        semantic_properties_table.to_excel("data/semantic_properties.xlsx")

    # todo: handle instances of classes listed in the schema.org JSON data
    # todo: save jsondata_slot and jsonschema_slot to file
    # todo: create other (dummy) slots
    # todo: bundle the slots to packages
    # todo: create semantic properties (instances of Category:Property)
    # todo: persist semantic property names and their mapping to schema.org properties

    # Output of the print statements:
    # Number of properties: 1461
    # Number of properties without rangeIncludes attribute (potentially deprecated):  1
    # Number of object properties: 1897
    # Number of semantic properties: 1448
    # todo: why are there less semantic properties than object properties?

    # todo: discussion: rangeIncludes can have multiple entries or just one (list of dicts or dict) -> how to handle?
    #  * Can the same be used for entries in each property filed?
    #  * Can a field or a list be validated against one or multiple entries in rangeIncludes?
    # todo: discussion: is the handling of @type in schema.org transferable to OO-LD?
# Line before last line of the document
