// Copyright 2023 The MediaPipe Authors.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//      http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos");

let handLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;

// Before we can use HandLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
const createHandLandmarker = async () => {
  console.log("Loading MediaPipe model...");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU"
    },
    runningMode: runningMode,
    numHands: 2
  });
  console.log("Model loaded successfully!");
  demosSection.classList.remove("invisible");
};
createHandLandmarker();

/********************************************************************
// Demo 1: Grab a bunch of images from the page and detection them
// upon click.
********************************************************************/

// In this demo, we have put all our clickable images in divs with the
// CSS class 'detectionOnClick'. Lets get all the elements that have
// this class.
const imageContainers = document.getElementsByClassName("detectOnClick");

// Now let's go through all of these and add a click event listener.
for (let i = 0; i < imageContainers.length; i++) {
  // Add event listener to the child element whichis the img element.
  imageContainers[i].children[0].addEventListener("click", handleClick);
}

// When an image is clicked, let's detect it and display results!
async function handleClick(event) {
  if (!handLandmarker) {
    console.log("Wait for handLandmarker to load before clicking!");
    return;
  }

  if (runningMode === "VIDEO") {
    runningMode = "IMAGE";
    await handLandmarker.setOptions({ runningMode: "IMAGE" });
  }
  // Remove all landmarks drawed before
  const allCanvas = event.target.parentNode.getElementsByClassName("canvas");
  for (var i = allCanvas.length - 1; i >= 0; i--) {
    const n = allCanvas[i];
    n.parentNode.removeChild(n);
  }

  // We can call handLandmarker.detect as many times as we like with
  // different image data each time. This returns a promise
  // which we wait to complete and then call a function to
  // print out the results of the prediction.
  const handLandmarkerResult = handLandmarker.detect(event.target);
  console.log(handLandmarkerResult.handednesses[0][0]);
  const canvas = document.createElement("canvas");
  canvas.setAttribute("class", "canvas");
  canvas.setAttribute("width", event.target.naturalWidth + "px");
  canvas.setAttribute("height", event.target.naturalHeight + "px");
  canvas.style =
    "left: 0px;" +
    "top: 0px;" +
    "width: " +
    event.target.width +
    "px;" +
    "height: " +
    event.target.height +
    "px;";

  event.target.parentNode.appendChild(canvas);
  const cxt = canvas.getContext("2d");
  for (const landmarks of handLandmarkerResult.landmarks) {
    drawConnectors(cxt, landmarks, HAND_CONNECTIONS, {
      color: "#00FF00",
      lineWidth: 5
    });
    drawLandmarks(cxt, landmarks, { color: "#FF0000", lineWidth: 1 });
  }
}

/********************************************************************
// Demo 2: Continuously grab image from webcam stream and detect it.
********************************************************************/

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

// Check if webcam access is supported.
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
  document.getElementById("downloadButton").addEventListener("click", downloadLandmarks);
  document.getElementById("downloadDistanceButton").addEventListener("click", downloadDistances);
  document.getElementById("downloadVideoButton").addEventListener("click", downloadVideo);
  document.getElementById("downloadFeaturesButton").addEventListener("click", downloadFeatures);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

async function startCountdown(seconds, callback) {
  for (let i = seconds; i > 0; i--) {
    countdownElement.innerText = i;
    countdownElement.style.display = "block";
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  countdownElement.innerText = "";
  countdownElement.style.display = "none";
  callback();
}

async function enableCam(event) {
  console.log("Button clicked, handLandmarker:", handLandmarker);
  
  if (!handLandmarker) {
    console.log("Wait! objectDetector not loaded yet.");
    alert("Please wait for the model to load. Check console for status.");
    return;
  }

  if (webcamRunning === true) {
    return;
  }

  webcamRunning = true;
  enableWebcamButton.disabled = true;
  enableWebcamButton.querySelector('.mdc-button__label').textContent = "STARTING...";
  document.getElementById("downloadButton").style.display = "none";
  document.getElementById("downloadDistanceButton").style.display = "none";
  document.getElementById("downloadVideoButton").style.display = "none";
  document.getElementById("downloadFeaturesButton").style.display = "none";
  document.getElementById("chartContainer").style.display = "none";
  document.getElementById("chartContainerLeft").style.display = "none";
  document.getElementById("rightFeaturesContainer").style.display = "none";
  document.getElementById("leftFeaturesContainer").style.display = "none";
  document.getElementById("recordingFpsDisplay").style.display = "none";
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  document.getElementById("videoContainer").style.display = "block";
  video.style.display = "block";
  canvasElement.style.display = "block";
  savedLandmarks = [];
  savedDistances = [];
  
  const constraints = { video: true };

  try {
    console.log("Requesting webcam...");
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    console.log("Webcam stream:", stream);
    video.srcObject = stream;
    
    video.onloadedmetadata = () => {
      console.log("Video metadata loaded");
      video.play();
      console.log("Starting countdown...");
      startCountdown(3, () => startRecording(predictWebcam, stream));
    };
  } catch (err) {
    console.error("Webcam error:", err);
    alert("Could not access webcam: " + err.message);
    webcamRunning = false;
    enableWebcamButton.disabled = false;
    enableWebcamButton.querySelector('.mdc-button__label').textContent = "START";
  }
}

async function startRecording(predictCallback, stream) {
  enableWebcamButton.querySelector('.mdc-button__label').textContent = "RECORDING...";
  isRecording = true;
  recordedChunks = [];
  
  mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp9' });
  
  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  };
  
  mediaRecorder.start(100);
  recordingStartTime = Date.now();
  
  predictCallback();
  
  countdownBarContainer.style.display = "block";
  countdownBar.style.width = "100%";
  fpsDisplay.style.display = "block";
  frameCount = 0;
  lastFpsTime = performance.now();
  
  for (let i = 10; i > 0; i--) {
    countdownElement.innerText = i;
    countdownElement.style.display = "block";
    countdownBar.style.width = (i * 10) + "%";
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  countdownElement.innerText = "";
  countdownElement.style.display = "none";
  countdownBarContainer.style.display = "none";
  fpsDisplay.style.display = "none";
  
  isRecording = false;
  webcamRunning = false;
  
  mediaRecorder.stop();
  
  await new Promise(resolve => {
    mediaRecorder.onstop = resolve;
  });
  
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(track => track.stop());
    video.srcObject = null;
  }
  
  video.style.display = "none";
  canvasElement.style.display = "none";
  document.getElementById("videoContainer").style.display = "none";
  
  enableWebcamButton.disabled = false;
  enableWebcamButton.querySelector('.mdc-button__label').textContent = "START";
  
  const downloadButton = document.getElementById("downloadButton");
  const downloadDistanceButton = document.getElementById("downloadDistanceButton");
  const downloadVideoButton = document.getElementById("downloadVideoButton");
  const downloadFeaturesButton = document.getElementById("downloadFeaturesButton");
  if (savedLandmarks.length > 0) {
    downloadButton.style.display = "inline-flex";
    downloadDistanceButton.style.display = "inline-flex";
    downloadVideoButton.style.display = "inline-flex";
    downloadFeaturesButton.style.display = "inline-flex";
    plotDistanceChart();
    
    const recordingFpsDisplay = document.getElementById("recordingFpsDisplay");
    const actualDuration = savedDistances.length > 1 ? (savedDistances[savedDistances.length - 1].timestamp - savedDistances[0].timestamp) / 1000 : 0;
    const recordingFps = actualDuration > 0 ? (savedDistances.length / actualDuration).toFixed(2) : 0;
    recordingFpsDisplay.innerHTML = `Recording FPS: <strong>${recordingFps}</strong> (${savedDistances.length} frames in ${actualDuration.toFixed(2)}s)`;
    recordingFpsDisplay.style.display = "block";
  } else {
    alert("No hand landmarks detected. Try again!");
  }
}

let lastVideoTime = -1;
let results = undefined;
let savedLandmarks = [];
let savedDistances = [];
let isRecording = false;
let countdownElement = document.getElementById("countdown");
let countdownBar = document.getElementById("countdownBar");
let countdownBarContainer = document.getElementById("countdownBarContainer");
let fpsDisplay = document.getElementById("fpsDisplay");
let frameCount = 0;
let lastFpsTime = 0;
let recordingStartTime = 0;
let mediaRecorder = null;
let recordedChunks = [];
console.log(video);

function calculateDistance(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;
  const lm4 = landmarks[4];
  const lm8 = landmarks[8];
  return Math.sqrt(Math.pow(lm8.x - lm4.x, 2) + Math.pow(lm8.y - lm4.y, 2));
}
async function predictWebcam() {
  if (!video.videoWidth || !video.videoHeight || video.readyState < 2) {
    if (webcamRunning) {
      window.requestAnimationFrame(predictWebcam);
    }
    return;
  }
  
  canvasElement.style.width = video.offsetWidth + "px";
  canvasElement.style.height = video.offsetHeight + "px";
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;
  
  // Now let's start detecting the stream.
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await handLandmarker.setOptions({ runningMode: "VIDEO" });
  }
  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    results = handLandmarker.detectForVideo(video, startTimeMs);
    if (isRecording) {
      const relativeTime = Date.now() - recordingStartTime;
      
      let leftHandLandmarks = null;
      let rightHandLandmarks = null;
      let leftHandDistance = null;
      let rightHandDistance = null;
      
      console.log("Results:", JSON.stringify(results));
      if (results.landmarks && results.handednesses) {
        console.log("Handednesses:", JSON.stringify(results.handednesses));
        for (let i = 0; i < results.landmarks.length; i++) {
          const handLandmarks = results.landmarks[i];
          const handedness = results.handednesses[i][0].displayName;
          const distance = calculateDistance(handLandmarks);
          
          if (handedness === "Left") {
            rightHandLandmarks = handLandmarks;
            rightHandDistance = distance;
          } else if (handedness === "Right") {
            leftHandLandmarks = handLandmarks;
            leftHandDistance = distance;
          }
        }
      }
      
      savedLandmarks.push({
        timestamp: relativeTime,
        videoTimestamp: video.currentTime,
        leftHand: {
          thumbTip: leftHandLandmarks ? { x: leftHandLandmarks[4].x, y: leftHandLandmarks[4].y } : null,
          indexFingerTip: leftHandLandmarks ? { x: leftHandLandmarks[8].x, y: leftHandLandmarks[8].y } : null
        },
        rightHand: {
          thumbTip: rightHandLandmarks ? { x: rightHandLandmarks[4].x, y: rightHandLandmarks[4].y } : null,
          indexFingerTip: rightHandLandmarks ? { x: rightHandLandmarks[8].x, y: rightHandLandmarks[8].y } : null
        }
      });
      savedDistances.push({
        timestamp: relativeTime,
        videoTimestamp: video.currentTime,
        leftHand: leftHandDistance,
        rightHand: rightHandDistance
      });
    }
  }
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  if (results.landmarks && results.handednesses) {
    for (let i = 0; i < results.landmarks.length; i++) {
      const landmarks = results.landmarks[i];
      const handedness = results.handednesses[i][0].displayName;
      const color = handedness === "Left" ? "#00FFFF" : "#00FF00";
      
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
        color: color,
        lineWidth: 5
      });
      drawLandmarks(canvasCtx, landmarks, { color: "#FF0000", lineWidth: 2 });
    }
  }
  canvasCtx.restore();
  
  frameCount++;
  const currentTime = performance.now();
  if (currentTime - lastFpsTime >= 1000) {
    const fps = Math.round(frameCount * 1000 / (currentTime - lastFpsTime));
    fpsDisplay.innerText = "FPS: " + fps;
    frameCount = 0;
    lastFpsTime = currentTime;
  }

  // Call this function again to keep predicting when the browser is ready.
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}

function downloadLandmarks() {
  if (savedLandmarks.length === 0) {
    console.log("No landmarks to download");
    return;
  }
  const dataStr = JSON.stringify(savedLandmarks, null, 2);
  const dataBlob = new Blob([dataStr], { type: "application/json" });
  const url = URL.createObjectURL(dataBlob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "hand_landmarks.json";
  link.click();
  URL.revokeObjectURL(url);
}

function downloadDistances() {
  if (savedDistances.length === 0) {
    console.log("No distances to download");
    return;
  }
  const dataStr = JSON.stringify(savedDistances, null, 2);
  const dataBlob = new Blob([dataStr], { type: "application/json" });
  const url = URL.createObjectURL(dataBlob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "hand_distances.json";
  link.click();
  URL.revokeObjectURL(url);
}


function downloadVideo() {
  if (recordedChunks.length === 0) {
    console.log("No video to download");
    return;
  }
  const videoBlob = new Blob(recordedChunks, { type: "video/webm" });
  const url = URL.createObjectURL(videoBlob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "hand_recording.webm";
  link.click();
  URL.revokeObjectURL(url);
}

let distanceChart = null;
let leftHandChart = null;

function findPeaks(data, options = {}) {
  const {
    height = 0.01,
    threshold = 0.005,
    distance = 2,
    prominence = 0.005,
    width = 2
  } = options;
  
  const peaks = [];
  const prominences = [];
  const leftBases = [];
  const rightBases = [];
  
  if (!data || data.length < 2) {
    return { peaks, prominences, leftBases, rightBases };
  }
  
  const n = data.length;
  
  for (let i = 1; i < n - 1; i++) {
    if (data[i] === null) continue;
    
    const prev = data[i - 1];
    const curr = data[i];
    const next = data[i + 1];
    
    if (curr === null) continue;
    const prevVal = prev !== null ? prev : curr;
    const nextVal = next !== null ? next : curr;
    
    if (curr > prevVal && curr > nextVal) {
      if (threshold > 0) {
        const minDiff = Math.min(
          Math.abs(curr - prevVal),
          Math.abs(curr - nextVal)
        );
        if (minDiff < threshold) continue;
      }
      
      if (height !== null) {
        const minHeight = typeof height === 'number' ? height : height[0];
        if (curr < minHeight) continue;
      }
      
      let leftBase = i;
      for (let j = i - 1; j >= 0; j--) {
        if (data[j] !== null && data[j] < curr) {
          leftBase = j;
          break;
        }
      }
      
      let rightBase = i;
      for (let j = i + 1; j < n; j++) {
        if (data[j] !== null && data[j] < curr) {
          rightBase = j;
          break;
        }
      }
      
      let leftMin = curr;
      for (let j = leftBase; j < i; j++) {
        if (data[j] !== null && data[j] < leftMin) {
          leftMin = data[j];
        }
      }
      
      let rightMin = curr;
      for (let j = i; j <= rightBase; j++) {
        if (data[j] !== null && data[j] < rightMin) {
          rightMin = data[j];
        }
      }
      
      const baseVal = Math.max(leftMin, rightMin);
      const prom = curr - baseVal;
      
      if (prominence !== null && prom < prominence) continue;
      
      peaks.push(i);
      prominences.push(prom);
      leftBases.push(leftBase);
      rightBases.push(rightBase);
    }
  }
  
  if (distance > 1 && peaks.length > 1) {
    const filteredPeaks = [];
    const filteredProminences = [];
    const filteredLeftBases = [];
    const filteredRightBases = [];
    
    filteredPeaks.push(peaks[0]);
    filteredProminences.push(prominences[0]);
    filteredLeftBases.push(leftBases[0]);
    filteredRightBases.push(rightBases[0]);
    
    for (let i = 1; i < peaks.length; i++) {
      const lastKept = filteredPeaks[filteredPeaks.length - 1];
      if (peaks[i] - lastKept >= distance) {
        filteredPeaks.push(peaks[i]);
        filteredProminences.push(prominences[i]);
        filteredLeftBases.push(leftBases[i]);
        filteredRightBases.push(rightBases[i]);
      } else if (prominences[i] > filteredProminences[filteredProminences.length - 1]) {
        filteredPeaks[filteredPeaks.length - 1] = peaks[i];
        filteredProminences[filteredProminences.length - 1] = prominences[i];
        filteredLeftBases[filteredLeftBases.length - 1] = leftBases[i];
        filteredRightBases[filteredRightBases.length - 1] = rightBases[i];
      }
    }
    
    return {
      peaks: filteredPeaks,
      prominences: filteredProminences,
      leftBases: filteredLeftBases,
      rightBases: filteredRightBases
    };
  }
  
  return { peaks, prominences, leftBases, rightBases };
}

function findValleys(peaks, data) {
  const valleys = [];
  
  if (!peaks || peaks.length < 2 || !data) {
    return valleys;
  }
  
  for (let i = 0; i < peaks.length - 1; i++) {
    const startIdx = peaks[i];
    const endIdx = peaks[i + 1];
    
    let minVal = Infinity;
    let minIdx = -1;
    
    for (let j = startIdx; j <= endIdx; j++) {
      if (data[j] !== null && data[j] < minVal) {
        minVal = data[j];
        minIdx = j;
      }
    }
    
    if (minIdx !== -1 && minIdx !== startIdx && minIdx !== endIdx) {
      valleys.push(minIdx);
    }
  }
  
  return valleys;
}

function detectPeaksAndValleys(data, options = {}) {
  const {
    threshold = 0.01,
    distance = null,
    prominence = null,
    height = null
  } = options;
  
  const peaksResult = findPeaks(data, {
    threshold,
    distance: distance || 1,
    prominence,
    height
  });
  
  const peaks = peaksResult.peaks;
  const valleys = findValleys(peaks, data);
  
  return {
    peaks: peaks,
    valleys: valleys,
    peakProminences: peaksResult.prominences,
    peakHeights: peaks.map(i => data[i])
  };
}

function createPointStyles(dataLength, peaks, valleys) {
  const styles = [];
  for (let i = 0; i < dataLength; i++) {
    if (peaks.includes(i)) {
      styles.push({ radius: 6, backgroundColor: '#FF0000', borderColor: '#FF0000' });
    } else if (valleys.includes(i)) {
      styles.push({ radius: 6, backgroundColor: '#0000FF', borderColor: '#0000FF' });
    } else {
      styles.push({ radius: 0, backgroundColor: 'transparent', borderColor: 'transparent' });
    }
  }
  return styles;
}

function plotDistanceChart() {
  if (savedDistances.length === 0) {
    console.log("No distances to plot");
    return;
  }
  
  const frameLabels = [];
  const rightHandDistances = [];
  const leftHandDistances = [];
  
  for (let i = 0; i < savedDistances.length; i++) {
    frameLabels.push(i + 1);
    rightHandDistances.push(savedDistances[i].rightHand);
    leftHandDistances.push(savedDistances[i].leftHand);
  }
  
  const rightPeaksValleys = detectPeaksAndValleys(rightHandDistances, {
    threshold: 0,
    distance: 1,
    prominence: 0,
    height: null
  });
  
  const leftPeaksValleys = detectPeaksAndValleys(leftHandDistances, {
    threshold: 0,
    distance: 1,
    prominence: 0,
    height: null
  });
  
  const rightPointStyles = createPointStyles(rightHandDistances.length, rightPeaksValleys.peaks, rightPeaksValleys.valleys);
  const leftPointStyles = createPointStyles(leftHandDistances.length, leftPeaksValleys.peaks, leftPeaksValleys.valleys);
  
  const chartContainer = document.getElementById("chartContainer");
  const chartContainerLeft = document.getElementById("chartContainerLeft");
  const ctxRight = document.getElementById("distanceChart").getContext("2d");
  const ctxLeft = document.getElementById("leftHandChart").getContext("2d");
  
  if (distanceChart) {
    distanceChart.destroy();
  }
  if (leftHandChart) {
    leftHandChart.destroy();
  }
  
  distanceChart = new Chart(ctxRight, {
    type: "line",
    data: {
      labels: frameLabels,
      datasets: [{
        label: "Right Hand Distance",
        data: rightHandDistances,
        borderColor: "#00FF00",
        backgroundColor: "rgba(0, 255, 0, 0.2)",
        borderWidth: 2,
        fill: true,
        tension: 0.1,
        pointRadius: rightPointStyles.map(s => s.radius),
        pointBackgroundColor: rightPointStyles.map(s => s.backgroundColor),
        pointBorderColor: rightPointStyles.map(s => s.borderColor),
        spanGaps: true
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: "Right Hand Distance Over Time (Red = Peak, Blue = Valley)",
          font: {
            size: 16
          }
        },
        legend: {
          display: true,
          position: "top"
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Frame"
          }
        },
        y: {
          title: {
            display: true,
            text: "Distance"
          },
          beginAtZero: true
        }
      }
    }
  });
  
  leftHandChart = new Chart(ctxLeft, {
    type: "line",
    data: {
      labels: frameLabels,
      datasets: [{
        label: "Left Hand Distance",
        data: leftHandDistances,
        borderColor: "#00FFFF",
        backgroundColor: "rgba(0, 255, 255, 0.2)",
        borderWidth: 2,
        fill: true,
        tension: 0.1,
        pointRadius: leftPointStyles.map(s => s.radius),
        pointBackgroundColor: leftPointStyles.map(s => s.backgroundColor),
        pointBorderColor: leftPointStyles.map(s => s.borderColor),
        spanGaps: true
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: "Left Hand Distance Over Time (Red = Peak, Blue = Valley)",
          font: {
            size: 16
          }
        },
        legend: {
          display: true,
          position: "top"
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Frame"
          }
        },
        y: {
          title: {
            display: true,
            text: "Distance"
          },
          beginAtZero: true
        }
      }
    }
  });
  
  chartContainer.style.display = "block";
  chartContainerLeft.style.display = "block";
  
  const rightTapCount = Math.min(rightPeaksValleys.peaks.length, rightPeaksValleys.valleys.length);
  const leftTapCount = Math.min(leftPeaksValleys.peaks.length, leftPeaksValleys.valleys.length);
  
  const actualDuration = savedDistances.length > 1 ? (savedDistances[savedDistances.length - 1].timestamp - savedDistances[0].timestamp) / 1000 : 0;
  
  const rightTapFreq = actualDuration > 0 ? rightTapCount / actualDuration : 0;
  const leftTapFreq = actualDuration > 0 ? leftTapCount / actualDuration : 0;
  
  let rightITI = 0;
  if (rightPeaksValleys.peaks.length > 1) {
    const itiSum = (rightPeaksValleys.peaks[rightPeaksValleys.peaks.length - 1] - rightPeaksValleys.peaks[0]) / (rightPeaksValleys.peaks.length - 1);
    rightITI = actualDuration > 0 ? (itiSum / savedDistances.length) * actualDuration * 1000 : 0;
  }
  
  let leftITI = 0;
  if (leftPeaksValleys.peaks.length > 1) {
    const itiSum = (leftPeaksValleys.peaks[leftPeaksValleys.peaks.length - 1] - leftPeaksValleys.peaks[0]) / (leftPeaksValleys.peaks.length - 1);
    leftITI = actualDuration > 0 ? (itiSum / savedDistances.length) * actualDuration * 1000 : 0;
  }
  
  document.getElementById("rightTapCount").textContent = rightTapCount;
  document.getElementById("rightTapFreq").textContent = rightTapFreq.toFixed(2);
  document.getElementById("rightITI").textContent = rightITI.toFixed(2);
  document.getElementById("leftTapCount").textContent = leftTapCount;
  document.getElementById("leftTapFreq").textContent = leftTapFreq.toFixed(2);
  document.getElementById("leftITI").textContent = leftITI.toFixed(2);
  
  document.getElementById("rightFeaturesContainer").style.display = "block";
  document.getElementById("leftFeaturesContainer").style.display = "block";
  
  window.currentFeatures = {
    rightHand: {
      totalTapCount: rightTapCount,
      tappingFrequencyHz: parseFloat(rightTapFreq.toFixed(2)),
      interTappingIntervalMs: parseFloat(rightITI.toFixed(2)),
      peakCount: rightPeaksValleys.peaks.length,
      valleyCount: rightPeaksValleys.valleys.length
    },
    leftHand: {
      totalTapCount: leftTapCount,
      tappingFrequencyHz: parseFloat(leftTapFreq.toFixed(2)),
      interTappingIntervalMs: parseFloat(leftITI.toFixed(2)),
      peakCount: leftPeaksValleys.peaks.length,
      valleyCount: leftPeaksValleys.valleys.length
    }
  };
}

function downloadFeatures() {
  if (!window.currentFeatures) {
    console.log("No features to download");
    return;
  }
  const dataStr = JSON.stringify(window.currentFeatures, null, 2);
  const dataBlob = new Blob([dataStr], { type: "application/json" });
  const url = URL.createObjectURL(dataBlob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "hand_features.json";
  link.click();
  URL.revokeObjectURL(url);
}