import json
import yaml
import boto3


class AwsSecretsManagerReader:
    """The AwsSecretsManagerReader class provides a simple interface
    for reading secrets from AWS Secret Manager. It takes care of the
    necessary API calls and authentication to retrieve secrets from the
    AWS service.
    To use the AwsSecretsManagerReader class, you will need to have AWS
    credentials or an AWS access token set up on your system. You can
    either set up your credentials as environment variables, or configure
    them using the AWS CLI.
    """

    def __init__(self, region_name="us-west-2", aws_profile_name=None):
        self.region_name = region_name
        self.aws_profile_name = aws_profile_name

    def get_secret(self, secret_name: str) -> dict:
        """Retrieves the value of the secret with the specified name from
        AWS Secret Manager.
        You can call the get_secret function and pass in the name of the secret
        you want to retrieve.
        A secret name in AWS can be associated with multiple key-value pairs.
        The function returns this information in a dictionary.
        Example:
            >>> secret_class.get_secret('prod/mysqlsecret')
            {
                "host":"sqlserver.com",
                "port": 3112,
                "user": "me",
                "pass": "123456"
            }
        """
        if self.aws_profile_name is not None:
            session = boto3.session.Session(profile_name=self.aws_profile_name)
        else:
            session = boto3.session.Session()
            
        client = session.client(
            service_name="secretsmanager", region_name=self.region_name
        )
        secrets_response = client.get_secret_value(SecretId=secret_name)

        secrets = json.loads(secrets_response["SecretString"])

        return secrets


