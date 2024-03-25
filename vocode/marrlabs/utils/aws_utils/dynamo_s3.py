"""
Utility code for interacting with DynamoDB and S3.
"""

import boto3
from botocore.exceptions import ClientError


class DynamoDBTable:
    """
    Represents a DynamoDB table. 

    Provides methods for creating and updating items (aka "rows" or "records").
    Assumes that AWS credentials are defined in the environment.

    TODO: make these methods async to reduce conversation latency.

    Example Usage:
    dynamodb_table = DynamoDBTable('YourTableName')
    new_item = {'id': '1', 'name': 'John Doe', 'age': 30}
    dynamodb_table.create_item(new_item)
    update_expression = 'SET age = :val1'
    expression_attribute_values = {':val1': 31}
    dynamodb_table.update_item({'id': '1'}, update_expression, expression_attribute_values)
    """
    def __init__(self, table_name, 
                 region_name='us-west-2', 
                 aws_profile_name=None):
        self.table_name = table_name
        self.aws_profile_name = aws_profile_name
        self.region_name = region_name
        
        if self.aws_profile_name is not None:
            session = boto3.session.Session(region_name=self.region_name,
                                    profile_name=self.aws_profile_name)
        else:
            session = boto3.session.Session(region_name=self.region_name)

        self.dynamodb = session.resource(service_name="dynamodb")

        self.table = self.dynamodb.Table(table_name)

    def create_item(self, item):
        try:
            response = self.table.put_item(Item=item)
            return response
        except ClientError as e:
            print(f"Failed to create item: {e}")
            return None

    def update_item(self, key, update_expression, expression_attribute_values):
        try:
            response = self.table.update_item(
                Key=key,
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_attribute_values,
                ReturnValues="UPDATED_NEW"
            )
            return response
        except ClientError as e:
            print(f"Failed to update item: {e}")
            return None

    def scan_table(self):
        # Scan the table 
        # Note: For large tables, consider using pagination
        return self.table.scan()