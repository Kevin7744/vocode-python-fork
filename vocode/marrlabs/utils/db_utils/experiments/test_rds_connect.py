#!/usr/bin/env python3
"""
This script tests the SQLAlchemy connection to a PostgreSQL database running in Amazon RDS
using credentials from AWS Secrets Manager.

Setup:
1. Get the database host from AWS console > RDS > Databases > <database> > 
   Connectivity & security > Endpoint
2. Get the database port from AWS console > RDS > Databases > <database> > 
   Connectivity & security > Port (default is 5432)
3. Get the database name from a team member. Traditionally, the name of the initial database is "postgres".
4. Get the database user from AWS console > RDS > Databases > <database> > 
   Connectivity & security > Master username.
   Alternatively, get the database user from a team member.
5. Ensure your local environment has the AWS credentials configured to access the AWS Secrets Manager secret.
   See https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html
6. On the security group for the RDS database, ensure that the inbound rules allow access from your IP address
   through the port specified in step 2.

Example test run:
➜  vocode-python git:(main) ✗ pwd
/Users/royseto/marrlabs/code/vocode-python
➜  vocode-python git:(main) ✗ poetry shell
Spawning shell within /Users/royseto/Library/Caches/pypoetry/virtualenvs/vocode-6X-vHCDy-py3.11

➜  vocode-python git:(main) ✗ emulate bash -c '. /Users/royseto/Library/Caches/pypoetry/virtualenvs/vocode-6X-vHCDy-py3.11/bin/activate'
(vocode-py3.11) ➜  vocode-python git:(main) ✗ cd vocode/utils/db_utils/experiments
(vocode-py3.11) ➜  experiments git:(main) ✗ ./test_rds_connect.py
Connection successful, result: 1
(vocode-py3.11) ➜  experiments git:(main) ✗
"""

import os

from sqlalchemy import create_engine, text
from vocode.utils.aws_utils.aws_secret_manager import AwsSecretsManagerReader


def test_sqlalchemy_connection():
    """
    Test the SQLAlchemy connection to a PostgreSQL database using credentials
    from AWS Secrets Manager.
    """
    # Update these details as per your database configuration
    host = "twilio-serverless.cluster-ckwkku9mctzw.us-west-2.rds.amazonaws.com"
    port = "5432"
    database_name = "postgres"
    user = "marrlabs"

    RDS_SECRET_KEY = "rds-db-credentials/cluster-XDVCTA3MGIC5DY6FGGNUAXTAOM/marrlabs/1705525305936"
    RDS_PASSWORD_KEY_STR = "password"
    secret_manager = AwsSecretsManagerReader("us-west-2", aws_profile_name="marrlabsmfa")
    rds_secrets = secret_manager.get_secret(RDS_SECRET_KEY)
    password = rds_secrets.get(RDS_PASSWORD_KEY_STR)

    try:
        # Create the database URI
        database_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database_name}"
        engine = create_engine(database_uri)

        # Test connection
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print(f"Connection successful, result: {result.scalar()}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    test_sqlalchemy_connection()
