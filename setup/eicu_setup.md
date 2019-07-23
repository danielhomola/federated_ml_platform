# Setting up EICU on EC2

## EC2

- go to the london AWS subspace (top right corner)
- spin up an Ubuntu 18.04 Server version
- Choose a t2.large instance
- add 100GB SSD
- create an elastic IP and pair it with
- copy the private key you create in the process to ~/.ssh/ on your comp
- create ~/.ssh/config with

```
Host aws
    HostName ec2-3-9-28-102.eu-west-2.compute.amazonaws.com
    User ubuntu
    IdentityFile ~/.ssh/aws_uk_laptop.pem
```

- This allows to ssh to the EC2 with simply `ssh aws`

## Get the DB working

- `sudo apt-get update`
- `sudo reboot`
- `sudo apt-get install make`
- `sudo apt-get install postgresql`
- `sudo service postgresql start`
- get the code of the [repo](https://github.com/MIT-LCP/eicu-code) `git clone https://github.com/mit-lcp/eicu-code.git`
- follow the instructions at the [build page]( https://github.com/mit-lcp/eicu-code.git)
- once you're past the `make initialize` part, change `etc/postgresql/10/main/pg_hba.conf` from `local  all  all  peer` to `local  all  all md5`, then run `sudo service postgresql restart`
- then you can build the db with `make eicu-gz datadir=<DATA_PATH>`.

## Connect to DB from PyCharm

- on the DB tab add PostGreSQL with ssh connection 
- matching your elastic IP, 
- ubuntu as username and 
- adding th .pem file that was downloaded when you created the instance
- in the general tab, use eicu for database, user, pass
- in the schemas tab choose eicu_crd

## EICU dataset

- [website](https://eicu-crd.mit.edu/)
- columns are defined [here](https://eicu-crd.mit.edu/eicutables/careplancareprovider/)
- [publication](https://www.nature.com/articles/sdata2018178)
