# --------------------------------------------------------------------------------------
# variables for connecting to our EC2 hosted eICU PostgreSQL DB
# --------------------------------------------------------------------------------------

class SSHInfoEicu(object):
    """
    Info to connect to our AWS EC2 with eICU DB.
    """
    host = ("ec2-3-9-28-102.eu-west-2.compute.amazonaws.com", 22)
    ssh_private_key = "~/.ssh/aws_uk_laptop.pem"
    ssh_username = "ubuntu"
    remote_bind_address = ("localhost", 5432)


class DBInfoEicu(object):
    """
    Info to connect to the eICU PostgreSQL DB.
    """
    db_name = db_user = db_password = 'eicu'


# --------------------------------------------------------------------------------------
# variables for connecting to the 3 proto hospitals at aws
# --------------------------------------------------------------------------------------


class Hospitals(object):
    """
    Info to connect to our AWS EC2 with websocket server running on them.
    """
    h1_name = "h1"
    h1_host = "ec2-35-178-147-241.eu-west-2.compute.amazonaws.com"
    h1_port = 8777

    h2_name = "h2"
    h2_host = "ec2-35-178-10-220.eu-west-2.compute.amazonaws.com"
    h2_port = 8777

    h3_name = "h3"
    h3_host = "ec2-3-9-230-244.eu-west-2.compute.amazonaws.com"
    h3_port = 8777
