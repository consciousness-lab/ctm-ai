#!/bin/bash
#
# Manage WebArena docker containers per service/category.
#
# Usage:
#   ./webarena_docker.sh start shopping         # start shopping container
#   ./webarena_docker.sh stop gitlab reddit      # stop gitlab and reddit
#   ./webarena_docker.sh create map              # create map containers
#   ./webarena_docker.sh remove shopping_admin   # remove shopping_admin container
#   ./webarena_docker.sh patch gitlab            # patch gitlab config
#   ./webarena_docker.sh load shopping           # load shopping docker image
#   ./webarena_docker.sh setup shopping gitlab   # full setup: remove + create + start + patch
#   ./webarena_docker.sh setup all               # full setup everything
#   ./webarena_docker.sh status                  # show running status
#   ./webarena_docker.sh serve_homepage          # start homepage flask server (foreground)
#   ./webarena_docker.sh serve_reset             # start reset server (foreground)
#
# Categories: shopping, shopping_admin, reddit, gitlab, wikipedia, map, all

set -e

# ---------------------------------------------------------------------------
# Configuration — adjust to match your environment
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBARENA_SETUP_DIR="${SCRIPT_DIR}/../../webarena-setup/webarena"
OSM_COMPOSE_DIR="${WEBARENA_SETUP_DIR}/openstreetmap-website"

# Source vars (ports, hostname, URLs, ARCHIVES_LOCATION)
source "${WEBARENA_SETUP_DIR}/00_vars.sh"

# Container names (must match 03_docker_create_containers.sh)
SHOPPING_CONTAINER="shopping"
SHOPPING_ADMIN_CONTAINER="shopping_admin"
REDDIT_CONTAINER="forum"
GITLAB_CONTAINER="gitlab"
WIKIPEDIA_CONTAINER="wikipedia"
MAP_CONTAINERS=("openstreetmap-website-db-1" "openstreetmap-website-web-1")

# Image names
declare -A IMAGE_NAME=(
    [shopping]="shopping_final_0712"
    [shopping_admin]="shopping_admin_final_0719"
    [reddit]="postmill-populated-exposed-withimg"
    [gitlab]="gitlab-populated-final-port8023"
    [wikipedia]="ghcr.io/kiwix/kiwix-serve:3.3.0"
    [map_db]="openstreetmap-website-db"
    [map_web]="openstreetmap-website-web"
)

# Archive files
declare -A ARCHIVE_FILE=(
    [shopping]="shopping_final_0712.tar"
    [shopping_admin]="shopping_admin_final_0719.tar"
    [reddit]="postmill-populated-exposed-withimg.tar"
    [gitlab]="gitlab-populated-final-port8023.tar"
    [map_db]="openstreetmap-website-db.tar.gz"
    [map_web]="openstreetmap-website-web.tar.gz"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
is_running() {
    docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null | grep -q true
}

container_exists() {
    docker inspect "$1" &>/dev/null
}

image_loaded() {
    docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${1}:"
}

wait_secs() {
    local label="$1" seconds="$2"
    echo -n "  Waiting ${seconds}s for ${label}..."
    sleep "$seconds"
    echo " done"
}

# ===========================================================================
# LOAD — load docker images from archives
# ===========================================================================
load_image() {
    local img="$1" archive="$2"
    if image_loaded "$img"; then
        echo "  [skip] image ${img} already loaded"
    else
        local path="${ARCHIVES_LOCATION}/${archive}"
        if [ ! -f "$path" ]; then
            echo "  [error] archive not found: ${path}"
            return 1
        fi
        echo "  Loading ${img} from ${path}..."
        docker load --input "$path"
    fi
}

load_shopping()       { echo "Loading shopping image...";       load_image "${IMAGE_NAME[shopping]}" "${ARCHIVE_FILE[shopping]}"; }
load_shopping_admin() { echo "Loading shopping_admin image..."; load_image "${IMAGE_NAME[shopping_admin]}" "${ARCHIVE_FILE[shopping_admin]}"; }
load_reddit()         { echo "Loading reddit image...";         load_image "${IMAGE_NAME[reddit]}" "${ARCHIVE_FILE[reddit]}"; }
load_gitlab()         { echo "Loading gitlab image...";         load_image "${IMAGE_NAME[gitlab]}" "${ARCHIVE_FILE[gitlab]}"; }
load_map()            {
    echo "Loading map images..."
    load_image "${IMAGE_NAME[map_db]}" "${ARCHIVE_FILE[map_db]}"
    load_image "${IMAGE_NAME[map_web]}" "${ARCHIVE_FILE[map_web]}"
    # Extract openstreetmap archive if needed
    if [ ! -d "${WEBARENA_SETUP_DIR}/openstreetmap-website" ]; then
        echo "  Extracting openstreetmap archive..."
        tar -xzf "${ARCHIVES_LOCATION}/openstreetmap-website.tar.gz" -C "${WEBARENA_SETUP_DIR}"
    else
        echo "  [skip] openstreetmap archive already extracted"
    fi
}
load_wikipedia() {
    echo "Loading wikipedia..."
    # wikipedia uses a public image, just ensure the zim file is in place
    local wiki_dir="${WEBARENA_SETUP_DIR}/wiki"
    local zim="wikipedia_en_all_maxi_2022-05.zim"
    if [ ! -f "${wiki_dir}/${zim}" ]; then
        mkdir -p "$wiki_dir"
        if [ -f "${ARCHIVES_LOCATION}/${zim}" ]; then
            echo "  Copying wikipedia archive..."
            cp "${ARCHIVES_LOCATION}/${zim}" "${wiki_dir}/"
        else
            echo "  [error] ${zim} not found in ${ARCHIVES_LOCATION}"
            return 1
        fi
    else
        echo "  [skip] wikipedia zim already in place"
    fi
}

# ===========================================================================
# CREATE — create containers (must be loaded first)
# ===========================================================================
create_shopping() {
    echo "Creating shopping container..."
    if container_exists "$SHOPPING_CONTAINER"; then
        echo "  [skip] ${SHOPPING_CONTAINER} already exists"
    else
        docker create --name "$SHOPPING_CONTAINER" -p "${SHOPPING_PORT}:80" "${IMAGE_NAME[shopping]}"
        echo "  [created] ${SHOPPING_CONTAINER}"
    fi
}

create_shopping_admin() {
    echo "Creating shopping_admin container..."
    if container_exists "$SHOPPING_ADMIN_CONTAINER"; then
        echo "  [skip] ${SHOPPING_ADMIN_CONTAINER} already exists"
    else
        docker create --name "$SHOPPING_ADMIN_CONTAINER" -p "${SHOPPING_ADMIN_PORT}:80" "${IMAGE_NAME[shopping_admin]}"
        echo "  [created] ${SHOPPING_ADMIN_CONTAINER}"
    fi
}

create_reddit() {
    echo "Creating reddit container..."
    if container_exists "$REDDIT_CONTAINER"; then
        echo "  [skip] ${REDDIT_CONTAINER} already exists"
    else
        docker create --name "$REDDIT_CONTAINER" -p "${REDDIT_PORT}:80" "${IMAGE_NAME[reddit]}"
        echo "  [created] ${REDDIT_CONTAINER}"
    fi
}

create_gitlab() {
    echo "Creating gitlab container..."
    if container_exists "$GITLAB_CONTAINER"; then
        echo "  [skip] ${GITLAB_CONTAINER} already exists"
    else
        docker create --name "$GITLAB_CONTAINER" -p "${GITLAB_PORT}:${GITLAB_PORT}" \
            "${IMAGE_NAME[gitlab]}" /opt/gitlab/embedded/bin/runsvdir-start \
            --env "GITLAB_PORT=${GITLAB_PORT}"
        echo "  [created] ${GITLAB_CONTAINER}"
    fi
}

create_wikipedia() {
    echo "Creating wikipedia container..."
    if container_exists "$WIKIPEDIA_CONTAINER"; then
        echo "  [skip] ${WIKIPEDIA_CONTAINER} already exists"
    else
        docker create --name "$WIKIPEDIA_CONTAINER" \
            --volume="${WEBARENA_SETUP_DIR}/wiki/:/data" \
            -p "${WIKIPEDIA_PORT}:80" \
            "${IMAGE_NAME[wikipedia]}" wikipedia_en_all_maxi_2022-05.zim
        echo "  [created] ${WIKIPEDIA_CONTAINER}"
    fi
}

create_map() {
    echo "Creating map containers..."
    if container_exists "${MAP_CONTAINERS[0]}"; then
        echo "  [skip] map containers already exist"
    else
        # Copy templates and configure
        cd "${WEBARENA_SETUP_DIR}"
        cp openstreetmap-templates/docker-compose.yml openstreetmap-website/docker-compose.yml
        cp openstreetmap-templates/leaflet.osm.js openstreetmap-website/vendor/assets/leaflet/leaflet.osm.js
        cp openstreetmap-templates/fossgis_osrm.js openstreetmap-website/app/assets/javascripts/index/directions/fossgis_osrm.js

        # Sed in-place portability
        if [[ "$OSTYPE" == "darwin"* ]]; then
            SED_INPLACE=(-i '')
        else
            SED_INPLACE=(-i)
        fi

        OSM_TILE_SERVER_URL="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        OSM_GEOCODING_SERVER_URL="https://nominatim.openstreetmap.org/"
        OSM_ROUTING_SERVER_URL="https://routing.openstreetmap.de"
        OSM_CAR_SUFFIX="/routed-car"
        OSM_BIKE_SUFFIX="/routed-bike"
        OSM_FOOT_SUFFIX="/routed-foot"

        cd openstreetmap-website/
        sed "${SED_INPLACE[@]}" "s|MAP_PORT|${MAP_PORT}|g" docker-compose.yml
        sed "${SED_INPLACE[@]}" "s|url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'|url: '${OSM_TILE_SERVER_URL}'|g" ./vendor/assets/leaflet/leaflet.osm.js
        sed "${SED_INPLACE[@]}" "s|nominatim_url:.*|nominatim_url: \"$OSM_GEOCODING_SERVER_URL\"|g" ./config/settings.yml
        sed "${SED_INPLACE[@]}" "s|fossgis_osrm_url:.*|fossgis_osrm_url: \"$OSM_ROUTING_SERVER_URL\"|g" ./config/settings.yml
        sed "${SED_INPLACE[@]}" "s|__OSMCarSuffix__|${OSM_CAR_SUFFIX}|g" ./app/assets/javascripts/index/directions/fossgis_osrm.js
        sed "${SED_INPLACE[@]}" "s|__OSMBikeSuffix__|${OSM_BIKE_SUFFIX}|g" ./app/assets/javascripts/index/directions/fossgis_osrm.js
        sed "${SED_INPLACE[@]}" "s|__OSMFootSuffix__|${OSM_FOOT_SUFFIX}|g" ./app/assets/javascripts/index/directions/fossgis_osrm.js

        docker compose create
        echo "  [created] openstreetmap"
        cd "$SCRIPT_DIR"
    fi
}

# ===========================================================================
# START — start existing containers
# ===========================================================================
start_container() {
    local name="$1" wait_sec="${2:-0}"
    if is_running "$name"; then
        echo "  [skip] ${name} is already running"
    else
        docker start "$name"
        echo "  [started] ${name}"
        [ "$wait_sec" -gt 0 ] && wait_secs "$name" "$wait_sec"
    fi
}

start_shopping()       { echo "Starting shopping...";       start_container "$SHOPPING_CONTAINER" 10; }
start_shopping_admin() { echo "Starting shopping_admin..."; start_container "$SHOPPING_ADMIN_CONTAINER" 10; }
start_reddit()         { echo "Starting reddit...";         start_container "$REDDIT_CONTAINER" 10; }
start_gitlab()         { echo "Starting gitlab...";         start_container "$GITLAB_CONTAINER" 30; }
start_wikipedia()      { echo "Starting wikipedia...";      start_container "$WIKIPEDIA_CONTAINER" 5; }
start_map() {
    echo "Starting map..."
    if is_running "${MAP_CONTAINERS[0]}" && is_running "${MAP_CONTAINERS[1]}"; then
        echo "  [skip] map containers already running"
    else
        (cd "$OSM_COMPOSE_DIR" && docker compose start)
        echo "  [started] openstreetmap"
        wait_secs "openstreetmap" 15
    fi
}

# ===========================================================================
# STOP — stop running containers
# ===========================================================================
stop_one() {
    local name="$1"
    if is_running "$name"; then
        docker stop "$name"
        echo "  [stopped] ${name}"
    else
        echo "  [skip] ${name} is not running"
    fi
}

stop_shopping()       { echo "Stopping shopping...";       stop_one "$SHOPPING_CONTAINER"; }
stop_shopping_admin() { echo "Stopping shopping_admin..."; stop_one "$SHOPPING_ADMIN_CONTAINER"; }
stop_reddit()         { echo "Stopping reddit...";         stop_one "$REDDIT_CONTAINER"; }
stop_gitlab()         { echo "Stopping gitlab...";         stop_one "$GITLAB_CONTAINER"; }
stop_wikipedia()      { echo "Stopping wikipedia...";      stop_one "$WIKIPEDIA_CONTAINER"; }
stop_map() {
    echo "Stopping map..."
    if [ -d "$OSM_COMPOSE_DIR" ]; then
        (cd "$OSM_COMPOSE_DIR" && docker compose stop)
        echo "  [stopped] openstreetmap"
    else
        for c in "${MAP_CONTAINERS[@]}"; do stop_one "$c"; done
    fi
}

# ===========================================================================
# REMOVE — stop + remove containers
# ===========================================================================
remove_one() {
    local name="$1"
    if container_exists "$name"; then
        docker stop "$name" 2>/dev/null || true
        docker rm "$name"
        echo "  [removed] ${name}"
    else
        echo "  [skip] ${name} does not exist"
    fi
}

remove_shopping()       { echo "Removing shopping...";       remove_one "$SHOPPING_CONTAINER"; }
remove_shopping_admin() { echo "Removing shopping_admin..."; remove_one "$SHOPPING_ADMIN_CONTAINER"; }
remove_reddit()         { echo "Removing reddit...";         remove_one "$REDDIT_CONTAINER"; }
remove_gitlab()         { echo "Removing gitlab...";         remove_one "$GITLAB_CONTAINER"; }
remove_wikipedia()      { echo "Removing wikipedia...";      remove_one "$WIKIPEDIA_CONTAINER"; }
remove_map() {
    echo "Removing map..."
    if [ -d "$OSM_COMPOSE_DIR" ]; then
        (cd "$OSM_COMPOSE_DIR" && docker compose down 2>/dev/null || true)
        echo "  [removed] openstreetmap"
    else
        for c in "${MAP_CONTAINERS[@]}"; do remove_one "$c"; done
    fi
}

# ===========================================================================
# PATCH — configure containers after start (ports, URLs, etc.)
# ===========================================================================
patch_shopping() {
    echo "Patching shopping..."
    docker exec "$SHOPPING_CONTAINER" /var/www/magento2/bin/magento setup:store-config:set \
        --base-url="http://${PUBLIC_HOSTNAME}:${SHOPPING_PORT}"
    docker exec "$SHOPPING_CONTAINER" mysql -u magentouser -pMyPassword magentodb -e \
        "UPDATE core_config_data SET value='http://${PUBLIC_HOSTNAME}:${SHOPPING_PORT}/' WHERE path = 'web/secure/base_url';"
    docker exec "$SHOPPING_CONTAINER" /var/www/magento2/bin/magento cache:flush
    echo "  [patched] shopping"
}

patch_shopping_admin() {
    echo "Patching shopping_admin..."
    docker exec "$SHOPPING_ADMIN_CONTAINER" php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
    docker exec "$SHOPPING_ADMIN_CONTAINER" php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0
    docker exec "$SHOPPING_ADMIN_CONTAINER" /var/www/magento2/bin/magento setup:store-config:set \
        --base-url="http://${PUBLIC_HOSTNAME}:${SHOPPING_ADMIN_PORT}"
    docker exec "$SHOPPING_ADMIN_CONTAINER" mysql -u magentouser -pMyPassword magentodb -e \
        "UPDATE core_config_data SET value='http://${PUBLIC_HOSTNAME}:${SHOPPING_ADMIN_PORT}/' WHERE path = 'web/secure/base_url';"
    docker exec "$SHOPPING_ADMIN_CONTAINER" /var/www/magento2/bin/magento cache:flush
    echo "  [patched] shopping_admin"
}

patch_reddit() {
    echo "Patching reddit..."
    docker exec "$REDDIT_CONTAINER" sed -i \
        -e 's/^pm.max_children = .*/pm.max_children = 32/' \
        -e 's/^pm.start_servers = .*/pm.start_servers = 10/' \
        -e 's/^pm.min_spare_servers = .*/pm.min_spare_servers = 5/' \
        -e 's/^pm.max_spare_servers = .*/pm.max_spare_servers = 20/' \
        -e 's/^;pm.max_requests = .*/pm.max_requests = 500/' \
        /usr/local/etc/php-fpm.d/www.conf
    docker exec "$REDDIT_CONTAINER" supervisorctl restart php-fpm
    echo "  [patched] reddit"
}

patch_gitlab() {
    echo "Patching gitlab..."
    docker exec "$GITLAB_CONTAINER" sed -i \
        "s|^external_url.*|external_url 'http://${PUBLIC_HOSTNAME}:${GITLAB_PORT}'|" \
        /etc/gitlab/gitlab.rb
    docker exec "$GITLAB_CONTAINER" bash -c \
        "printf '\n\npuma[\"worker_processes\"] = 4' >> /etc/gitlab/gitlab.rb"
    docker exec "$GITLAB_CONTAINER" gitlab-ctl reconfigure
    echo "  [patched] gitlab"
}

patch_wikipedia() {
    echo "Patching wikipedia..."
    echo "  [skip] wikipedia needs no patch"
}

patch_map() {
    echo "Patching map..."
    docker exec openstreetmap-website-web-1 bin/rails db:migrate RAILS_ENV=development
    echo "  [patched] map"
}

# ===========================================================================
# SETUP — full lifecycle: remove → create → start → patch
# ===========================================================================
setup_shopping()       { remove_shopping;       create_shopping;       start_shopping;       patch_shopping; }
setup_shopping_admin() { remove_shopping_admin;  create_shopping_admin; start_shopping_admin; patch_shopping_admin; }
setup_reddit()         { remove_reddit;          create_reddit;         start_reddit;         patch_reddit; }
setup_gitlab()         { remove_gitlab;          create_gitlab;         start_gitlab;         patch_gitlab; }
setup_wikipedia()      { remove_wikipedia;       create_wikipedia;      start_wikipedia;      patch_wikipedia; }
setup_map()            { remove_map;             create_map;            start_map;            patch_map; }

# ===========================================================================
# SERVE — homepage and reset server (run in foreground)
# ===========================================================================
serve_homepage() {
    echo "Starting homepage server on port ${HOMEPAGE_PORT}..."
    cd "${WEBARENA_SETUP_DIR}/webarena-homepage"
    cp templates/index.backup templates/index.html

    if [[ "$OSTYPE" == "darwin"* ]]; then
        SED_INPLACE=(-i '')
    else
        SED_INPLACE=(-i)
    fi
    sed "${SED_INPLACE[@]}" "s|SHOPPING_URL|${SHOPPING_URL}|g" templates/index.html
    sed "${SED_INPLACE[@]}" "s|SHOPPING_ADMIN_URL|${SHOPPING_ADMIN_URL}|g" templates/index.html
    sed "${SED_INPLACE[@]}" "s|GITLAB_URL|${GITLAB_URL}|g" templates/index.html
    sed "${SED_INPLACE[@]}" "s|REDDIT_URL|${REDDIT_URL}|g" templates/index.html
    sed "${SED_INPLACE[@]}" "s|MAP_URL|${MAP_URL}|g" templates/index.html
    sed "${SED_INPLACE[@]}" "s|WIKIPEDIA_URL|${WIKIPEDIA_URL}|g" templates/index.html

    flask run --host=0.0.0.0 --port="$HOMEPAGE_PORT"
}

serve_reset() {
    echo "Starting reset server on port ${RESET_PORT}..."
    cd "${WEBARENA_SETUP_DIR}/reset_server"
    python server.py --port "${RESET_PORT}" 2>&1 | tee -a server.log
}

# ===========================================================================
# STATUS
# ===========================================================================
show_status() {
    printf "%-30s %s\n" "CONTAINER" "STATUS"
    printf "%-30s %s\n" "------------------------------" "----------"
    for name in "$SHOPPING_CONTAINER" "$SHOPPING_ADMIN_CONTAINER" "$REDDIT_CONTAINER" \
                "$GITLAB_CONTAINER" "$WIKIPEDIA_CONTAINER" "${MAP_CONTAINERS[@]}"; do
        if is_running "$name"; then
            printf "%-30s %s\n" "$name" "running"
        elif container_exists "$name"; then
            printf "%-30s %s\n" "$name" "stopped"
        else
            printf "%-30s %s\n" "$name" "not created"
        fi
    done
}

# ===========================================================================
# Dispatch
# ===========================================================================
VALID_ACTIONS="load create start stop remove patch setup"

do_action() {
    local action="$1" category="$2"
    case "$category" in
        shopping)       ${action}_shopping ;;
        shopping_admin) ${action}_shopping_admin ;;
        reddit)         ${action}_reddit ;;
        gitlab)         ${action}_gitlab ;;
        wikipedia)      ${action}_wikipedia ;;
        map)            ${action}_map ;;
        all)
            ${action}_shopping
            ${action}_shopping_admin
            ${action}_reddit
            ${action}_gitlab
            ${action}_wikipedia
            ${action}_map
            ;;
        *)
            echo "Unknown category: ${category}"
            echo "Valid: shopping, shopping_admin, reddit, gitlab, wikipedia, map, all"
            exit 1
            ;;
    esac
}

# ===========================================================================
# Main
# ===========================================================================
ACTION="${1:-}"
shift || true

case "$ACTION" in
    status)
        show_status
        exit 0
        ;;
    serve_homepage)
        serve_homepage
        exit 0
        ;;
    serve_reset)
        serve_reset
        exit 0
        ;;
    load|create|start|stop|remove|patch|setup)
        if [ $# -eq 0 ]; then
            echo "Error: specify at least one category (or 'all')"
            exit 1
        fi
        for cat in "$@"; do
            do_action "$ACTION" "$cat"
        done
        ;;
    *)
        cat <<'USAGE'
Usage: webarena_docker.sh <action> [category ...]

Actions:
  load      Load docker images from archive files
  create    Create containers (images must be loaded)
  start     Start existing containers
  stop      Stop running containers
  remove    Stop and remove containers
  patch     Configure containers (ports, URLs, etc.)
  setup     Full lifecycle: remove + create + start + patch
  status    Show container status
  serve_homepage   Start homepage flask server (foreground)
  serve_reset      Start reset server (foreground)

Categories: shopping, shopping_admin, reddit, gitlab, wikipedia, map, all

Examples:
  webarena_docker.sh setup shopping           # full setup shopping only
  webarena_docker.sh setup all                # full setup everything
  webarena_docker.sh start gitlab reddit      # start gitlab and reddit
  webarena_docker.sh stop map                 # stop only map
  webarena_docker.sh patch shopping_admin     # reconfigure shopping_admin
  webarena_docker.sh status                   # check what's running
USAGE
        exit 1
        ;;
esac

echo ""
echo "Done."
