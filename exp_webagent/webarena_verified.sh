#!/bin/bash

# ==========================================
# Global Configuration (Change once, applies everywhere)
# ==========================================
HOST_IP="localhost"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/setup"

PORT_SHOPPING=7770
PORT_SHOPPING_ADMIN=7780
PORT_REDDIT=9999
PORT_GITLAB=8023
PORT_WIKIPEDIA=8888
PORT_MAP=3030
PORT_HOMEPAGE=4399

PORT_SHOPPING_DBG=7771
PORT_SHOPPING_ADMIN_DBG=7781
PORT_REDDIT_DBG=9998
PORT_GITLAB_DBG=8024
PORT_WIKIPEDIA_DBG=8889
PORT_MAP_DBG=3031

# Array of all container names
CONTAINERS=(
  "webarena-verified-shopping"
  "webarena-verified-shopping_admin"
  "webarena-verified-reddit"
  "webarena-verified-gitlab"
  "webarena-verified-wikipedia"
  "webarena-verified-map"
)

# Site short names (same order as CONTAINERS), for: ./script start shopping map
SITES=(shopping shopping_admin reddit gitlab wikipedia map)

# Resolve optional site names to container list; no args = all containers
TARGET_CONTAINERS=()
resolve_sites() {
  TARGET_CONTAINERS=()
  if [[ $# -eq 0 ]]; then
    TARGET_CONTAINERS=("${CONTAINERS[@]}")
    return
  fi
  for s in "$@"; do
    for i in "${!SITES[@]}"; do
      if [[ "${SITES[i]}" == "$s" ]]; then
        TARGET_CONTAINERS+=("${CONTAINERS[i]}")
        break
      fi
    done
  done
}

# ==========================================
# Core Functions
# ==========================================

# Print environment variables required for evaluation (always full list, unchanged)
print_exports() {
  echo -e "\n📋 Please copy the following environment variables to your terminal, or add them to your evaluation startup script:"
  cat << EOF
export WA_SHOPPING="http://${HOST_IP}:${PORT_SHOPPING}"
export WA_SHOPPING_ADMIN="http://${HOST_IP}:${PORT_SHOPPING_ADMIN}/admin"
export WA_REDDIT="http://${HOST_IP}:${PORT_REDDIT}/forums/all"
export WA_GITLAB="http://${HOST_IP}:${PORT_GITLAB}/explore"
export WA_WIKIPEDIA="http://${HOST_IP}:${PORT_WIKIPEDIA}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="http://${HOST_IP}:${PORT_MAP}"
export WA_HOMEPAGE="http://${HOST_IP}:${PORT_HOMEPAGE}"
export PW_EXTRA_HEADERS=""
EOF
  echo -e "--------------------------------------------------\n"
}

# Init a single site by index (0=shopping .. 5=map)
init_site_by_index() {
  local i=$1
  case $i in
    0) docker run -d --name ${CONTAINERS[0]} -p ${PORT_SHOPPING}:80 -p ${PORT_SHOPPING_DBG}:8877 am1n3e/webarena-verified-shopping ;;
    1) docker run -d --name ${CONTAINERS[1]} -p ${PORT_SHOPPING_ADMIN}:80 -p ${PORT_SHOPPING_ADMIN_DBG}:8877 am1n3e/webarena-verified-shopping_admin ;;
    2) docker run -d --name ${CONTAINERS[2]} -p ${PORT_REDDIT}:80 -p ${PORT_REDDIT_DBG}:8877 am1n3e/webarena-verified-reddit ;;
    3) docker run -d --name ${CONTAINERS[3]} -p ${PORT_GITLAB}:8023 -p ${PORT_GITLAB_DBG}:8877 am1n3e/webarena-verified-gitlab ;;
    4) docker run -d --name ${CONTAINERS[4]} -p ${PORT_WIKIPEDIA}:8080 -p ${PORT_WIKIPEDIA_DBG}:8874 -v ${DATA_DIR}:/data:ro am1n3e/webarena-verified-wikipedia ;;
    5)
      webarena-verified env setup init --site map --data-dir ${DATA_DIR}
      echo "⚠️ Ensure you have already run: webarena-verified env setup init before this"
      docker run -d --name ${CONTAINERS[5]} -p ${PORT_MAP}:3000 -p ${PORT_MAP_DBG}:8877 \
        -v webarena_verified_map_tile_db:/data/database \
        -v webarena_verified_map_routing_car:/data/routing/car \
        -v webarena_verified_map_routing_bike:/data/routing/bike \
        -v webarena_verified_map_routing_foot:/data/routing/foot \
        -v webarena_verified_map_nominatim_db:/data/nominatim/postgres \
        -v webarena_verified_map_nominatim_flatnode:/data/nominatim/flatnode \
        -v webarena_verified_map_website_db:/var/lib/postgresql/14/main \
        -v webarena_verified_map_tiles:/data/tiles \
        -v webarena_verified_map-style:/data/style \
        am1n3e/webarena-verified-map
      ;;
    *) echo "Unknown site index: $i" ; exit 1 ;;
  esac
}

# 1. Init: Create and start containers (all or specified sites)
init_all() {
  shift  # drop "init"
  resolve_sites "$@"
  if [[ ${#TARGET_CONTAINERS[@]} -eq ${#CONTAINERS[@]} ]]; then
    echo "🚀 [INIT] Creating and starting all WebArena containers for the first time..."
  else
    echo "🚀 [INIT] Creating and starting selected containers: ${TARGET_CONTAINERS[*]}"
  fi
  for c in "${TARGET_CONTAINERS[@]}"; do
    for i in "${!CONTAINERS[@]}"; do
      if [[ "${CONTAINERS[i]}" == "$c" ]]; then
        init_site_by_index "$i"
        break
      fi
    done
  done
  echo "✅ Initialization complete!"
  print_exports
}

# 2. Start: Wake up existing containers (all or specified sites)
start_all() {
  shift
  resolve_sites "$@"
  if [[ ${#TARGET_CONTAINERS[@]} -eq 0 ]]; then
    echo "No valid sites specified."
    exit 1
  fi
  if [[ ${#TARGET_CONTAINERS[@]} -eq ${#CONTAINERS[@]} ]]; then
    echo "🟢 [START] Waking up all containers..."
  else
    echo "🟢 [START] Waking up: ${TARGET_CONTAINERS[*]}"
  fi
  docker start "${TARGET_CONTAINERS[@]}"
  echo "✅ Startup complete!"
  print_exports
}

# 3. Stop: Stop containers (all or specified sites)
stop_all() {
  shift
  resolve_sites "$@"
  if [[ ${#TARGET_CONTAINERS[@]} -eq 0 ]]; then
    echo "No valid sites specified."
    exit 1
  fi
  echo "🔴 [STOP] Stopping: ${TARGET_CONTAINERS[*]}"
  docker stop "${TARGET_CONTAINERS[@]}"
  echo "✅ Stop complete!"
}

# 4. Restart: Restart containers (all or specified sites)
restart_all() {
  shift
  resolve_sites "$@"
  if [[ ${#TARGET_CONTAINERS[@]} -eq 0 ]]; then
    echo "No valid sites specified."
    exit 1
  fi
  echo "🔄 [RESTART] Restarting: ${TARGET_CONTAINERS[*]}"
  docker restart "${TARGET_CONTAINERS[@]}"
  echo "✅ Restart complete!"
  print_exports
}

# 5. Remove: Force destroy containers (all or specified sites)
remove_all() {
  shift
  resolve_sites "$@"
  if [[ ${#TARGET_CONTAINERS[@]} -eq 0 ]]; then
    echo "No valid sites specified."
    exit 1
  fi
  echo "🗑️ [REMOVE] Force removing: ${TARGET_CONTAINERS[*]} (data volumes are kept intact)..."
  docker rm -f "${TARGET_CONTAINERS[@]}"
  echo "✅ Removal complete!"
}

# 6. Reset: One-click environment reset (all or specified sites)
reset_all() {
  shift
  resolve_sites "$@"
  if [[ ${#TARGET_CONTAINERS[@]} -eq 0 ]]; then
    echo "No valid sites specified."
    exit 1
  fi
  echo "☢️ [RESET] Destroying and recreating: ${TARGET_CONTAINERS[*]}"
  docker rm -f "${TARGET_CONTAINERS[@]}"
  echo "--------------------------------------------------"
  for c in "${TARGET_CONTAINERS[@]}"; do
    for i in "${!CONTAINERS[@]}"; do
      if [[ "${CONTAINERS[i]}" == "$c" ]]; then
        init_site_by_index "$i"
        break
      fi
    done
  done
  echo "✅ Reset complete!"
  print_exports
}

# ==========================================
# Command-line Argument Routing
# ==========================================
case "$1" in
  init)    init_all "$@" ;;
  start)   start_all "$@" ;;
  stop)    stop_all "$@" ;;
  restart) restart_all "$@" ;;
  remove)  remove_all "$@" ;;
  reset)   reset_all "$@" ;;
  *)
    echo "Usage: $0 {init|start|stop|restart|remove|reset} [site ...]"
    echo "  site: optional, one or more of: ${SITES[*]} (default: all)"
    echo ""
    echo "  init    - Create and start the environment for the first time (runs docker run)"
    echo "  start   - Wake up existing environment (runs docker start)"
    echo "  stop    - Pause execution to free resources (runs docker stop)"
    echo "  restart - Restart containers"
    echo "  remove  - Force destroy containers (clean up environment)"
    echo "  reset   - Destroy and recreate (ideal for restoring a clean state after running an Agent)"
    echo ""
    echo "Examples:"
    echo "  $0 start                    # start all"
    echo "  $0 start shopping map      # start only shopping and map"
    echo "  $0 init wikipedia          # init only wikipedia"
    exit 1
    ;;
esac