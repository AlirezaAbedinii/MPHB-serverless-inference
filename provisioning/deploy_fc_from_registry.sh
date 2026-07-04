#!/bin/bash

# ---- CONFIGURATION ----
# Replace this with your own registry base, e.g. registry.cn-hangzhou.cr.aliyuncs.com/your-namespace
REGISTRY="<your-registry>.<region>.personal.cr.aliyuncs.com/<namespace>/<repo>"

# Example mapping of Docker images to Function Compute function names
# Format: docker_image=function_name
declare -A FUNCTION_IMAGE_MAP=(
    # ["cf0tto1w1"]="cfrom0to10worker1"
    # ["cf0tto1w2"]="cfrom0to10worker2"
    # ["cf0tto1w3"]="cfrom0to10worker3"
    # ["cf0tto1w4"]="cfrom0to10worker4"
    
    # ["cf11ttow1"]="cfrom11to15worker1"
    # ["cf11ttow2"]="cfrom11to15worker2"
    # ["cf11ttow3"]="cfrom11to15worker3"
    # ["cf11ttow4"]="cfrom11to15worker4"

    # ["cf16ttow1"]="cfrom16to19worker1"
    # ["cf16ttow2"]="cfrom16to19worker2"
    # ["cf16ttow3"]="cfrom16to19worker3"
    # ["cf16ttow4"]="cfrom16to19worker4"

    # ["cf21ttow1"]="cfrom21to21worker1"
    # ["cf21ttow2"]="cfrom21to21worker2"
    # ["cf21ttow3"]="cfrom21to21worker3"
    # ["cf21ttow4"]="cfrom21to21worker4"

    # ["cmaster"]="cmaster"


    # ["f0tto1w1log"]="from0to10worker1"
    # ["f0tto1w2log"]="from0to10Worker2"
    # ["f0tto1w3log"]="from0to10Worker3log"
    # ["f0tto1w4log"]="from0to10Worker4log"
    
    # ["cf11ttow1"]="from11to15Worker1log"
    # ["cf11ttow2"]="from11to15Worker2log"
    # ["cf11ttow3"]="from11to15Worker3log"
    # ["cf11ttow4"]="from11to15Worker4log"

    # ["cf16ttow1"]="from16to19Worker1log"
    # ["cf16ttow2"]="from16to19Worker2log"
    # ["cf16ttow3"]="from16to19Worker3log"
    # ["cf16ttow4"]="from16to19Worker4log"

    # ["cf21ttow1"]="from21to21Worker1log"
    # ["cf21ttow2"]="from21to21Worker2log"
    # ["cf21ttow3"]="from21to21Worker3log"
    # ["cf21ttow4"]="from21to21Worker4log"

    # ["cwrnf0tto1w1"]="wcfrom0to17Worker1"
    # ["cwrnf0tto1w2"]="wcfrom0to17Worker2"
    # ["cwrnf0tto1w3"]="wcfrom0to17Worker3"
    # ["cwrnf0tto1w4"]="wcfrom0to17Worker4"
    
    # ["cwrnf18ttow1"]="wcfrom18to23Worker1"
    # ["cwrnf18ttow2"]="wcfrom18to23Worker2"
    # ["cwrnf18ttow3"]="wcfrom18to23Worker3"
    # ["cwrnf18ttow4"]="wcfrom18to23Worker4"

    # ["cwrnf24ttow1"]="wcfrom24to29Worker1"
    # ["cwrnf24ttow2"]="wcfrom24to29Worker2"
    # ["cwrnf24ttow3"]="wcfrom24to29Worker3"
    # ["cwrnf24ttow4"]="wcfrom24to29Worker4"

    # ["cf11ttow1"]="from11to15Worker1log"
    # ["cf11ttow2"]="from11to15Worker2log"
    # ["cf11ttow3"]="from11to15Worker3log"
    # ["cf11ttow4"]="from11to15Worker4log"

    # ["cf16ttow1"]="from16to19Worker1log"
    # ["cf16ttow2"]="from16to19Worker2log"
    # ["cf16ttow3"]="from16to19Worker3log"
    # ["cf16ttow4"]="from16to19Worker4log"

    # ["wgf0ttow1"]="wgfrom0to17Worker1"
    # ["cf21ttow2"]="from21to21Worker2log"
    # ["cf21ttow3"]="from21to21Worker3log"
    # ["cf21ttow4"]="from21to21Worker4log"

    # ["wgf0tto1w1"]="wgfrom0to17Worker1"
    # ["wgf0ttow2"]="wgfrom0to17Worker2"
    # ["wgf0ttow3"]="wgfrom0to17Worker3"
    # ["wgf0ttow4"]="wgfrom0to17Worker4"
    
    # ["wgf18ttow1"]="wcfrom18to23Worker1"
    # ["wgf18ttow2"]="wgfrom18to23Worker2"
    # ["wgf18ttow3"]="wgfrom18to23Worker3"
    # ["wgf18ttow4"]="wgfrom18to23Worker4"

    # ["wgf24ttow1"]="wgfrom24to29Worker1"
    # ["wgf24ttow2"]="wgfrom24to29Worker2"
    # ["wgf24ttow3"]="wgfrom24to29Worker3"
    # ["wgf24ttow4"]="wgfrom24to29Worker4"
    

    ['cv2wrn5f18ttow1'] = 'from18to29Worker1'
    ['cv2wrn5f18ttow2'] = 'from18to29Worker2'
    # ['cv2wrn5f18ttow3'] = 'from18to29Worker3'
    # ['cv2wrn5f18ttow4'] = 'from18to29Worker4'

    # ['w5master'] = 'w5master'
)


# Region
REGION="us-east-1"  # or your region

# ------------ DEPLOY LOOP -------------
for IMAGE in "${!FUNCTION_IMAGE_MAP[@]}"; do
  FUNC_NAME="${FUNCTION_IMAGE_MAP[$IMAGE]}"
  FULL_IMAGE="${REGISTRY}:${IMAGE}"

  echo "🔁 Updating function: ${FUNC_NAME}"
  echo "📦 Image: ${FULL_IMAGE}"
#   aliyun fc PUT "/2021-04-06/functions/${FUNC}" --body "$(cat <<EOF
# {
#   "functionName": "${FUNC}",
#   "customContainerConfig": {
#     "image": "${IMAGE_URL}"
#   },
#   "runtime": "custom-container"
# }
# EOF
# )"
  aliyun fc PUT "/2023-03-30/functions/${FUNC_NAME}" --body "{
    \"functionName\": \"${FUNC_NAME}\",
    \"customContainerConfig\": {
      \"image\": \"${FULL_IMAGE}\"
    },
    \"runtime\": \"custom-container\"
  }"

  if [ $? -eq 0 ]; then
    echo "✅ Function ${FUNC_NAME} updated successfully"
  else
    echo "❌ Failed to update function ${FUNC_NAME} trying again"
    aliyun fc PUT "/2023-03-30/functions/${FUNC_NAME}" --body "{
    \"functionName\": \"${FUNC_NAME}\",
    \"customContainerConfig\": {
      \"image\": \"${FULL_IMAGE}\"
    },
    \"runtime\": \"custom-container\"
  }"
  fi

  echo "-----------------------------"
done