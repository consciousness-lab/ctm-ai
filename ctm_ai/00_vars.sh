#!/bin/bash

# PUBLIC_HOSTNAME=$(curl -s ifconfig.me)
PUBLIC_HOSTNAME="localhost"

# Change ports as desired
SHOPPING_PORT=8082
SHOPPING_ADMIN_PORT=8083
REDDIT_PORT=8080
GITLAB_PORT=9001
WIKIPEDIA_PORT=8081
MAP_PORT=443
HOMEPAGE_PORT=80
RESET_PORT=7565

# Original webarena ports
# SHOPPING_PORT=7770
# SHOPPING_ADMIN_PORT=7780
# REDDIT_PORT=9999
# GITLAB_PORT=8023
# WIKIPEDIA_PORT=8888
# MAP_PORT=3000
# HOMEPAGE_PORT=4399

# SHOPPING_URL="http://${PUBLIC_HOSTNAME}:${SHOPPING_PORT}"
# SHOPPING_ADMIN_URL="http://${PUBLIC_HOSTNAME}:${SHOPPING_ADMIN_PORT}/admin"
# REDDIT_URL="http://${PUBLIC_HOSTNAME}:${REDDIT_PORT}/forums/all"
# #GITLAB_URL="http://${PUBLIC_HOSTNAME}:${GITLAB_PORT}/explore"
# WIKIPEDIA_URL="http://${PUBLIC_HOSTNAME}:${WIKIPEDIA_PORT}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
# MAP_URL="http://${PUBLIC_HOSTNAME}:${MAP_PORT}"

export WA_SHOPPING="http://${PUBLIC_HOSTNAME}:${SHOPPING_PORT}"
export WA_SHOPPING_ADMIN="http://${PUBLIC_HOSTNAME}:${SHOPPING_ADMIN_PORT}/admin"
export WA_REDDIT="http://${PUBLIC_HOSTNAME}:${REDDIT_PORT}/forums/all"
export WA_GITLAB="http://${PUBLIC_HOSTNAME}:${GITLAB_PORT}/explore"
export WA_WIKIPEDIA="http://${PUBLIC_HOSTNAME}:${WIKIPEDIA_PORT}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="http://${PUBLIC_HOSTNAME}:${MAP_PORT}"
export WA_HOMEPAGE="http://${BASE_URL}:${HOMEPAGE_PORT}"

# export WA_SHOPPING="$BASE_URL:8082/"
# export WA_SHOPPING_ADMIN="$BASE_URL:8083/admin"
# export WA_REDDIT="$BASE_URL:8080"
# export WA_GITLAB="$BASE_URL:9001"
# export WA_WIKIPEDIA="$BASE_URL:8081/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
# export WA_MAP="$BASE_URL:443"
# export WA_HOMEPAGE="$BASE_URL:80"

# download the archives from the webarena instructions
# https://github.com/web-arena-x/webarena/tree/main/environment_docker
# Download the additional openstreetmap docker files from Zenodo (see README)
#  - shopping_final_0712.tar
#  - shopping_admin_final_0719.tar
#  - postmill-populated-exposed-withimg.tar
#  - gitlab-populated-final-port8023.tar
#  - openstreetmap-website-db.tar.gz
#  - openstreetmap-website-web.tar.gz
#  - openstreetmap-website.tar.gz
#  - wikipedia_en_all_maxi_2022-05.zim

ARCHIVES_LOCATION="./"
