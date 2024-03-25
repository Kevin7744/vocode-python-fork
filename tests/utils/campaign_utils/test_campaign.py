import pytest
from vocode.utils.campaign_utils.campaign import (GenericCampaign, 
                                                  BusinessCampaign)

def test_generic_campaign_from_dict():
    data = {"campaignTypeId": "123", 
            "campaignType": "email", 
            "campaignParams": 
                {"param1": "value1"}}
    campaign = GenericCampaign.from_dict(data)

    assert campaign.campaign_type_id == "123"
    assert campaign.campaign_type == "email"
    assert campaign.campaign_params == {"param1": "value1"}

def test_business_campaign_from_dict():
    data = {
        "campaignTypeId": "123", 
        "campaignType": "email", 
        "campaignParams": {"param1": "value1"}, 
        "businessType": "retail"
    }
    campaign = BusinessCampaign.from_dict(data)

    assert campaign.campaign_type_id == "123"
    assert campaign.campaign_type == "email"
    assert campaign.campaign_params == {"param1": "value1"}
    assert campaign.business_type == "retail"

# define a fixture function: a special kind of function 
# that sets up a resource needed for the tests below
# and then cleans up after them. 
@pytest.fixture
def business_campaign():
    data = {
        "campaignTypeId": "123", 
        "campaignType": "email", 
        "campaignParams": {"Prompt": "Test prompt", 
                           "DynParams": {"dyn_param1": "Value1",
                                         "dyn_param2": {"Default": "value2"}
                                         }
                           }, 
        "businessType": "retail"
    }
    return BusinessCampaign.from_dict(data)

def test_business_campaign_dyn_params(business_campaign):
    assert business_campaign.dyn_params == {"dyn_param1": "Value1",
                                            "dyn_param2": "value2"}

def test_business_campaign_prompt(business_campaign):
    assert business_campaign.prompt == "test prompt"

def test_business_campaign_llm_prompt(business_campaign):
    business_campaign.campaign_params["LLMPrompt"] = "LLM Test prompt <dyn_param1>, <dyn_param2>"
    assert business_campaign.llm_prompt == "llm test prompt value1, value2"

def test_business_campaign_name(business_campaign):
    business_campaign.campaign_params["CampaignNameTemplate"] = "Campaign <businesstype> <campaigntype>"
    assert business_campaign.campaign_name == "campaign retail email"

def test_business_campaign_name_with_dyn_params(business_campaign):
    business_campaign.campaign_params["CampaignNameTemplate"] = "Campaign <businesstype> <dyn_param1>"
    assert business_campaign.campaign_name == "campaign retail value1"

def test_business_campaign_id(business_campaign):
    assert business_campaign.campaign_id is not None

def test_business_campaign_first_prompt_for_biz(business_campaign):
    result = business_campaign.first_prompt_for_biz("MyBusiness")
    expected = "test prompt".replace("<businessname>", "MyBusiness")
    assert result == expected

def test_business_campaign_llm_prompt_for_biz(business_campaign):
    business_campaign.campaign_params["LLMPrompt"] = "LLM <businessname> prompt"
    result = business_campaign.llm_prompt_for_biz("MyBusiness")
    assert result == "llm MyBusiness prompt"