# --------------------------------------------------------------------------------------
# variables for connecting to our EC2 hosted eICU PostgreSQL DB
# --------------------------------------------------------------------------------------


class SSHInfoEicu(object):
    """
    Info to connect to our AWS EC2 with eICU DB.
    """
    host = ("ec2-3-9-28-102.eu-west-2.compute.amazonaws.com", 22)
    ssh_private_key = "/home/daniel/.ssh/aws_uk_laptop.pem"
    ssh_username = "ubuntu"
    remote_bind_address = ("localhost", 5432)


class DBInfoEicu(object):
    """
    Info to connect to the eICU PostgreSQL DB.
    """
    db_name = db_user = db_password = 'eicu'
