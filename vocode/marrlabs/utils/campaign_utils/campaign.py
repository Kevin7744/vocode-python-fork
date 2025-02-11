import hashlib
import random 

BUSINESS_NAME_HOLDER = "<businessname>"
BUSINESS_TYPE_HOLDER = "<businesstype>"
CAMPAIGN_TYPE_HOLDER = "<campaigntype>"

class GenericCampaign:
    """
    GenericCampaign:
        This class serves as a base for more specific campaign types.

        Properties:
            campaign_type_id: Unique identifier for campaign type.
            campaign_type: The type of campaign (e.g "Sales call", "Survey").
            campaign parameters: A dict containing various campain paramaters.

        also abtract methods:
            prompt: Needs to be implemented to return opening prompt for the call
            campaign_name: Needs to be implemented by subclasses to return the full campaign name
            
    """"
    def __init__(self, campaign_type_id, campaign_type, 
                       campaign_params):
        self.campaign_type_id = campaign_type_id
        self.campaign_type = campaign_type
        self.campaign_params = campaign_params

    @classmethod
    def from_dict(cls, data):
        campaign_type_id = data.get("campaignTypeId")
        campaign_type = data.get("campaignType")
        campaign_params = data.get("campaignParams")
        return cls(campaign_type_id, campaign_type, 
                    campaign_params)

    def prompt(self):
        pass
    
    def campaign_name(self):
        pass
    
    def campaign_id(self):
        pass

class BusinessCampaign(GenericCampaign):
    """
    BusinessCampaign:
        Inherits from GenericCampaign
        adds functionality to specific to campaings targettting businesses.

        additional property:
            business_type: The type of business the campaign is targettting ("Restaurant", "Retail")

        abstract methods:
            prompt: retrieves the opening prompt from campaign parameters, potentially selecting randomly if multiple options are provided. Replaces the placeholders with actual business name.
            llm_prompt: retrieves the llm prompt from campaign parameters, replacing placeholders with the business name
            campaing_name: retrieves the campaign name template from campaign parameters, 
                           It replaces placeholders like <businessname>, <businesstype>, and <campaigntype> with actual values. 
                           Additionally, it calculates a unique campaign ID based on the generated campaign name using SHA-256 hashing.

        helper methods:
            _get_prompt(): Handles retrieving the opening prompt, considering the possibility of multiple options.
            dyn_params: A property to access and set dynamic parameters from the campaign configuration. It ensures that dynamic parameters are defined as a dictionary.
            _replace_dyn_params(): Replaces placeholders in text (like prompts) with values from dynamic parameters. It can perform a case-insensitive replacement.
            first_prompt_for_biz(): Generates the opening prompt specifically for a business by replacing the <businessname> placeholder.
            llm_prompt_for_biz(): Generates the LLM prompt for a business by replacing the <businessname> placeholder.
            llm_prompt_for_sales(): Generates the LLM prompt specifically for a sales call, replacing placeholders for business name and potentially agent availability.
            
    """
    def __init__(self, campaign_type_id, campaign_type, 
                       campaign_params, business_type):
        super().__init__(campaign_type_id, 
                         campaign_type, 
                         campaign_params)
        self.business_type = business_type
        self.__campaign_name = None
        self.__campaign_id = None
        self.__prompt = None
        self.__llm_prompt = None
        self.__dyn_params = None
        
    @classmethod
    def from_dict(cls, data):
        campaign_type_id = data.get("campaignTypeId")
        campaign_type = data.get("campaignType")
        campaign_params = data.get("campaignParams")
        business_type = data.get("businessType")
        return cls(campaign_type_id, campaign_type, 
                   campaign_params, business_type)

    def _get_prompt(self):
        if "MultiplePrompts" in self.campaign_params:
            return random.choice(self.campaign_params["MultiplePrompts"])
        else:
            return self.campaign_params.get("Prompt", "")
            
    @property
    def dyn_params(self):
        if self.__dyn_params is None:
            self.__dyn_params = {}
            dyn_params = self.campaign_params.get("DynParams", {})
            for k, v in dyn_params.items():
                if isinstance(v, dict):
                    self.__dyn_params[k] = v.get("Default", "")
                else:
                    self.__dyn_params[k] = v
        return self.__dyn_params

    @dyn_params.setter
    def dyn_params(self, value):
        if not isinstance(value, dict):
            raise ValueError("dyn_params must be a dictionary")
        self.__dyn_params = value

    def _replace_dyn_params(self, text, lower_case=False):
        for key, value in self.dyn_params.items():  
            if lower_case:
                key = key.lower()
                text = text.lower()
            text = text.replace(f"<{key}>", value)
        return text

    @property
    def prompt(self):
        if self.__prompt is None:
            self.__prompt = self._get_prompt()
            self.__prompt = self._replace_dyn_params(self.__prompt,
                                                     lower_case=True)
        return self.__prompt

    @property
    def llm_prompt(self):
        if self.__llm_prompt is None:
            self.__llm_prompt = self.campaign_params.get("LLMPrompt", "")
            self.__llm_prompt = self._replace_dyn_params(self.__llm_prompt,
                                                         lower_case=True)
        return self.__llm_prompt

    @property
    def campaign_name(self):
        if self.__campaign_name is None:
            self.__campaign_name = self.campaign_params.get("CampaignNameTemplate", "")
            self.__campaign_name = self._replace_dyn_params(self.__campaign_name, 
                                                            lower_case=False)

            for (tag, value) in [(BUSINESS_TYPE_HOLDER, self.business_type),
                                 (CAMPAIGN_TYPE_HOLDER, self.campaign_type)]:
                self.__campaign_name = (self.__campaign_name
                                        .lower()
                                        .replace(tag, value) 
                                        )

        print(self.__campaign_name)
        return self.__campaign_name

    @property
    def campaign_id(self):
        if self.campaign_name is not None:
            self.__campaign_id = (hashlib.sha256(str(self.campaign_name)
                                                 .encode('utf-8'))
                                  .hexdigest()
                                  )
        return self.__campaign_id
        
    def first_prompt_for_biz(self, business_name):
        return self.prompt.lower().replace(BUSINESS_NAME_HOLDER, 
                                           business_name) 
    
    def llm_prompt_for_biz(self, business_name):
        return self.llm_prompt.lower().replace(BUSINESS_NAME_HOLDER, 
                                               business_name) 

    def llm_prompt_for_sales(self, business_name, agent_availability):
        sales_prompt = self.llm_prompt.lower().replace("<businessname>", 
                                                        business_name) 
        return sales_prompt.replace("<agentavailability>", 
                                     agent_availability)
