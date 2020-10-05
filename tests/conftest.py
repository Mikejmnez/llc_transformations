# # Import modules
# import subprocess


# # Download data
# def pytest_configure():
#     print("\nHello! I'm downloading test data.")

#     # Directory
#     Datadir = "/Users/Mikejmnez/llc_transformations/tests/"

#     # Download xmitgcm test
#     commands = [
#         "cd {}".format(Datadir),
#         "rm -fr Data",
#         "wget -v -O Data.tar.gz -L "
#         "https://livejohnshopkins-my.sharepoint.com/"
#         ":u:/g/personal/malmans2_jh_edu/"
#         "EVtxjAQL13tCt7dFHNrzsrwBuqdjDe3zrvrJ625YrjrF0g"
#         "?download=1",
#         "tar xvzf Data.tar.gz",
#         "rm -f Data.tar.gz",
#     ]
#     subprocess.call("&&".join(commands), shell=True)


