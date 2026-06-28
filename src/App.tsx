import React, { useState, useEffect, useRef, useMemo } from "react";
import { 
  Play, 
  Pause, 
  Video, 
  Cpu, 
  Layers, 
  Settings, 
  Download, 
  Copy, 
  Check, 
  FileCode, 
  Terminal, 
  Eye, 
  EyeOff, 
  HelpCircle, 
  Activity, 
  RefreshCw, 
  SlidersHorizontal, 
  ArrowRight, 
  BookOpen, 
  Sparkles, 
  AlertCircle,
  Upload
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { pythonFiles, PythonFile } from "./pythonCode";

// Define TypeScript types for internal tracker
interface TrackPoint {
  x: number;
  y: number;
}

interface SimulatedObject {
  id: number;
  classId: number;
  className: string;
  // Bounding box (absolute px values on 640x360 coordinates)
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  // Visual movement vectors
  targetX: number;
  targetY: number;
  speed: number;
  direction: number; // angle in radians
  color: string;
  history: TrackPoint[];
  status: "Active" | "Lost" | "Initializing";
  framesLost: number;
  confidence: number;
  lifetime: number;
}

// Predefined professional colors for classes
const CLASS_COLORS: { [key: string]: string } = {
  "Person": "#3b82f6", // Blue
  "Car": "#10b981",    // Green
  "Truck": "#f59e0b",  // Amber
  "Bicycle": "#8b5cf6",// Purple
  "Backpack": "#ec4899",// Pink
  "Object": "#14b8a6", // Teal
  "Hand": "#ef4444",   // Red
  "Face": "#06b6d4"    // Cyan
};

export default function App() {
  // Navigation State
  const [activeTab, setActiveTab] = useState<"dashboard" | "code" | "guide">("dashboard");

  // Code Hub state
  const [selectedFile, setSelectedFile] = useState<PythonFile>(pythonFiles[3]); // Default to main.py
  const [copied, setCopied] = useState<boolean>(false);

  // Configuration Sliders
  const [confThreshold, setConfThreshold] = useState<number>(0.25);
  const [maxAge, setMaxAge] = useState<number>(15);
  const [minHits, setMinHits] = useState<number>(3);
  const [targetFps, setTargetFps] = useState<number>(30);
  const [metricMode, setMetricMode] = useState<"iou" | "cosine">("iou");

  // Analytics overlays toggle
  const [showBoxes, setShowBoxes] = useState<boolean>(true);
  const [showIds, setShowIds] = useState<boolean>(true);
  const [showTrails, setShowTrails] = useState<boolean>(true);
  const [showKalmanPredict, setShowKalmanPredict] = useState<boolean>(true);
  const [showHudOverlay, setShowHudOverlay] = useState<boolean>(true);
  const [showGrid, setShowGrid] = useState<boolean>(false);

  // Simulation state
  const [streamSource, setStreamSource] = useState<"traffic" | "pedestrian" | "conveyor" | "webcam">("traffic");
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(30);
  const [gpuLoad, setGpuLoad] = useState<number>(45);

  // Webcam stream elements
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [webcamActive, setWebcamActive] = useState<boolean>(false);
  const [webcamError, setWebcamError] = useState<string | null>(null);
  const [useSimulatedWebcam, setUseSimulatedWebcam] = useState<boolean>(false);
  const [uploadedMediaUrl, setUploadedMediaUrl] = useState<string | null>(null);
  const [uploadedMediaType, setUploadedMediaType] = useState<"video" | "image" | null>(null);
  const [uploadedMediaName, setUploadedMediaName] = useState<string>("");
  const [isUploadedActive, setIsUploadedActive] = useState<boolean>(false);

  // Canvas context reference
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Statistics accumulators
  const [totalObjectsCount, setTotalObjectsCount] = useState<number>(142);
  const [classCounts, setClassCounts] = useState<{ [key: string]: number }>({
    "Car": 82,
    "Person": 39,
    "Truck": 14,
    "Bicycle": 7,
    "Object": 0
  });

  // Trackers collection state (for UI list rendering)
  const [activeTrackersList, setActiveTrackersList] = useState<SimulatedObject[]>([]);
  
  // Historical data for charts
  const [fpsHistory, setFpsHistory] = useState<number[]>(Array(30).fill(30));
  const [trackHistoryCount, setTrackHistoryCount] = useState<number[]>(Array(30).fill(5));

  // Simulation persistent reference variables (avoid React state trigger latency inside high-speed render loop)
  const stateRef = useRef<{
    objects: SimulatedObject[];
    nextId: number;
    frameCount: number;
    lastTime: number;
    prevFrameData: ImageData | null;
    webcamTrackerId: number;
  }>({
    objects: [],
    nextId: 101,
    frameCount: 0,
    lastTime: performance.now(),
    prevFrameData: null,
    webcamTrackerId: 1
  });

  // Handle stream source switching
  useEffect(() => {
    // Clear state
    stateRef.current.objects = [];
    stateRef.current.prevFrameData = null;
    setClassCounts({
      "Car": streamSource === "traffic" ? 82 : streamSource === "pedestrian" ? 14 : 0,
      "Person": streamSource === "traffic" ? 12 : streamSource === "pedestrian" ? 39 : 0,
      "Truck": streamSource === "traffic" ? 14 : 0,
      "Bicycle": streamSource === "traffic" ? 7 : streamSource === "pedestrian" ? 5 : 0,
      "Object": streamSource === "conveyor" ? 64 : 0
    });

    if (streamSource === "webcam") {
      setUseSimulatedWebcam(false);
      if (!isUploadedActive) {
        startWebcam();
      }
    } else {
      stopWebcam();
      setUseSimulatedWebcam(false);
      setIsUploadedActive(false);
      if (uploadedMediaUrl) {
        URL.revokeObjectURL(uploadedMediaUrl);
        setUploadedMediaUrl(null);
        setUploadedMediaType(null);
        setUploadedMediaName("");
      }
    }
  }, [streamSource]);

  // Webcam initiation
  const startWebcam = async () => {
    setWebcamError(null);
    setWebcamActive(false);
    try {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        // Use highly compatible flexible constraints to avoid OverconstrainedError or rejections
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { 
            width: { ideal: 640 }, 
            height: { ideal: 360 }, 
            facingMode: "user"
          },
          audio: false
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          
          // Use onloadedmetadata to ensure we play when ready, and catch any play promise rejections safely
          videoRef.current.onloadedmetadata = () => {
            if (videoRef.current) {
              videoRef.current.play()
                .then(() => {
                  console.log("Webcam stream started playing successfully.");
                })
                .catch(err => {
                  console.warn("Video play promise rejected, but stream is active:", err);
                });
            }
          };

          // Also try to call play immediately to trigger quick playback
          try {
            const playPromise = videoRef.current.play();
            if (playPromise !== undefined) {
              playPromise.catch(err => {
                console.warn("Immediate play promise rejected, waiting for metadata:", err);
              });
            }
          } catch (e) {
            console.warn("Immediate play threw synchronous exception:", e);
          }

          setWebcamActive(true);
        } else {
          // Fallback if Ref isn't ready in time
          setTimeout(() => {
            if (videoRef.current) {
              videoRef.current.srcObject = stream;
              videoRef.current.play().catch(() => {});
              setWebcamActive(true);
            } else {
              setWebcamError("Video rendering element not initialized.");
            }
          }, 150);
        }
      } else {
        setWebcamError("Camera capture APIs (navigator.mediaDevices.getUserMedia) are not supported or blocked in this browser/iframe context.");
      }
    } catch (err: any) {
      console.error("Webcam activation error:", err);
      setWebcamError("Camera access denied (Permission Denied). Security policies often block camera/mic capture inside embed frames. Please click 'Import Video or Photo' below to upload any file, use 'Simulated Camera', or click 'Open in Full Tab ↗' to bypass this restriction!");
    }
  };

  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setWebcamActive(false);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Revoke any previous URL
    if (uploadedMediaUrl) {
      URL.revokeObjectURL(uploadedMediaUrl);
    }

    const isVideo = file.type.startsWith("video/");
    const isImage = file.type.startsWith("image/");

    if (!isVideo && !isImage) {
      alert("Unsupported file type. Please upload a video or an image.");
      return;
    }

    const url = URL.createObjectURL(file);
    setUploadedMediaUrl(url);
    const mediaType = isVideo ? "video" : "image";
    setUploadedMediaType(mediaType);
    setUploadedMediaName(file.name);
    setIsUploadedActive(true);
    setWebcamActive(false);
    setUseSimulatedWebcam(false);
    setWebcamError(null);
    setStreamSource("webcam");

    if (isVideo) {
      stopWebcam();
      // Configure video element to play file source
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = null;
          videoRef.current.src = url;
          videoRef.current.loop = true;
          videoRef.current.muted = true;
          videoRef.current.playsInline = true;
          videoRef.current.play()
            .then(() => {
              setWebcamActive(true);
            })
            .catch(err => {
              console.warn("Error playing uploaded video file:", err);
              setWebcamActive(true);
            });
        }
      }, 150);
    } else {
      stopWebcam();
      // Set webcamActive to true once image is selected so the render loop triggers
      setWebcamActive(true);
    }
  };

  const clearUploadedMedia = () => {
    setIsUploadedActive(false);
    if (uploadedMediaUrl) {
      URL.revokeObjectURL(uploadedMediaUrl);
    }
    setUploadedMediaUrl(null);
    setUploadedMediaType(null);
    setUploadedMediaName("");
    setWebcamActive(false);
    // Restart default webcam
    startWebcam();
  };

  // Ensure webcam is released on unmount
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, []);

  // Copy code handler
  const copyToClipboard = () => {
    navigator.clipboard.writeText(selectedFile.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Trigger file download
  const downloadFile = (file: PythonFile) => {
    const element = document.createElement("a");
    const fileBlob = new Blob([file.code], { type: "text/plain" });
    element.href = URL.createObjectURL(fileBlob);
    element.download = file.name;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  // Generate initial simulation objects based on preset
  const generateSimulatedObjects = (preset: "traffic" | "pedestrian" | "conveyor", count: number) => {
    const arr: SimulatedObject[] = [];
    const baseId = stateRef.current.nextId;

    if (preset === "traffic") {
      const classes = ["Car", "Car", "Car", "Truck", "Bicycle"];
      for (let i = 0; i < count; i++) {
        const classStr = classes[i % classes.length];
        const isLeftToRight = Math.random() > 0.5;
        const color = CLASS_COLORS[classStr];
        
        // Stagger positions across horizontal lanes
        const yCoord = 140 + (i % 3) * 55 + Math.random() * 15;
        const xCoord = Math.random() * 640;
        const width = classStr === "Truck" ? 90 : classStr === "Bicycle" ? 30 : 65;
        const height = classStr === "Truck" ? 50 : classStr === "Bicycle" ? 50 : 38;

        arr.push({
          id: baseId + i,
          classId: classStr === "Car" ? 2 : classStr === "Truck" ? 7 : 1,
          className: classStr,
          x1: xCoord,
          y1: yCoord,
          x2: xCoord + width,
          y2: yCoord + height,
          targetX: isLeftToRight ? 700 : -100,
          targetY: yCoord,
          speed: 1.5 + Math.random() * 2.0,
          direction: isLeftToRight ? 0 : Math.PI,
          color: color,
          history: [],
          status: "Active",
          framesLost: 0,
          confidence: 0.82 + Math.random() * 0.16,
          lifetime: Math.floor(Math.random() * 200)
        });
      }
      stateRef.current.nextId += count;
    } else if (preset === "pedestrian") {
      const classes = ["Person", "Person", "Person", "Backpack"];
      for (let i = 0; i < count; i++) {
        const classStr = classes[i % classes.length];
        const angle = Math.random() * Math.PI * 2;
        const color = CLASS_COLORS[classStr];
        const x = Math.random() * 600;
        const y = 80 + Math.random() * 200;
        const w = classStr === "Backpack" ? 22 : 28;
        const h = classStr === "Backpack" ? 30 : 75;

        arr.push({
          id: baseId + i,
          classId: classStr === "Person" ? 0 : 24,
          className: classStr,
          x1: x,
          y1: y,
          x2: x + w,
          y2: y + h,
          targetX: x + Math.cos(angle) * 300,
          targetY: y + Math.sin(angle) * 300,
          speed: 0.6 + Math.random() * 0.7,
          direction: angle,
          color: color,
          history: [],
          status: "Active",
          framesLost: 0,
          confidence: 0.78 + Math.random() * 0.18,
          lifetime: Math.floor(Math.random() * 150)
        });
      }
      stateRef.current.nextId += count;
    } else if (preset === "conveyor") {
      for (let i = 0; i < count; i++) {
        const x = -80 - (i * 180);
        const y = 160;
        const color = CLASS_COLORS["Object"];
        arr.push({
          id: baseId + i,
          classId: 39, // bottle/object class
          className: "Object",
          x1: x,
          y1: y,
          x2: x + 40,
          y2: y + 60,
          targetX: 700,
          targetY: y,
          speed: 1.2,
          direction: 0,
          color: color,
          history: [],
          status: "Active",
          framesLost: 0,
          confidence: 0.95 + Math.random() * 0.04,
          lifetime: 0
        });
      }
      stateRef.current.nextId += count;
    }

    return arr;
  };

  // High performance Canvas Draw Loop
  useEffect(() => {
    let animationId: number;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    // Load initial objects if empty
    if (stateRef.current.objects.length === 0 && streamSource !== "webcam") {
      stateRef.current.objects = generateSimulatedObjects(streamSource as any, streamSource === "traffic" ? 5 : streamSource === "pedestrian" ? 6 : 4);
    }

    const renderLoop = () => {
      // Calculate delta time & target FPS capping
      const now = performance.now();
      const elapsed = now - stateRef.current.lastTime;
      const interval = 1000 / targetFps;

      if (!isPaused) {
        stateRef.current.frameCount++;

        // Render Background base depending on preset
        ctx.fillStyle = "#0d1527";
        ctx.fillRect(0, 0, 640, 360);

        if (streamSource === "traffic") {
          drawTrafficBackground(ctx);
        } else if (streamSource === "pedestrian") {
          drawPedestrianBackground(ctx);
        } else if (streamSource === "conveyor") {
          drawConveyorBackground(ctx);
        }

        // --- PIPELINE STEP A & B: DETECTION AND TRACKING SIMULATION ---
        if (streamSource === "webcam") {
          if (useSimulatedWebcam) {
            // Draw simulated office room or desk view with a moving subject!
            drawSimulatedOfficeCamera(ctx);
            // Run real-time webcam frame motion analysis and local tracker!
            processWebcamTracking(ctx);
          } else if (webcamActive && (videoRef.current || (uploadedMediaType === "image" && imgRef.current))) {
            if (uploadedMediaType === "image" && imgRef.current) {
              try {
                // Draw uploaded image
                ctx.drawImage(imgRef.current, 0, 0, 640, 360);
                // Run real-time webcam frame motion analysis and local tracker!
                processWebcamTracking(ctx);
              } catch (e) {
                // Image drawing/loading error
                ctx.fillStyle = "#1e293b";
                ctx.fillRect(40, 40, 560, 280);
                ctx.fillStyle = "#94a3b8";
                ctx.font = "14px Inter";
                ctx.fillText("Error loading uploaded image. Try another file.", 120, 180);
              }
            } else if (videoRef.current && (videoRef.current.readyState >= 2 || isUploadedActive)) {
              try {
                // Draw real video frame (webcam or uploaded file)
                ctx.drawImage(videoRef.current, 0, 0, 640, 360);
                // Run real-time webcam frame motion analysis and local tracker!
                processWebcamTracking(ctx);
              } catch (e) {
                // Draw camera fallback error
                ctx.fillStyle = "#1e293b";
                ctx.fillRect(40, 40, 560, 280);
                ctx.fillStyle = "#94a3b8";
                ctx.font = "14px Inter";
                ctx.fillText("Failed to capture video frames. Try another file or camera source.", 120, 180);
              }
            } else {
              // Video element is active but stream buffers are still decoding
              ctx.fillStyle = "#0f172a";
              ctx.fillRect(0, 0, 640, 360);
              
              ctx.fillStyle = "#38bdf8";
              ctx.shadowColor = "#38bdf8";
              ctx.shadowBlur = 6;
              ctx.font = "bold 15px Inter";
              ctx.fillText("DECODING VIDEO STREAM...", 210, 170);
              ctx.shadowBlur = 0;
 
              ctx.fillStyle = "#64748b";
              ctx.font = "11px JetBrains Mono";
              ctx.fillText("Establishing direct hardware media stream bindings...", 160, 205);
            }
          } else {
            // Camera not active state
            ctx.fillStyle = "#0f172a";
            ctx.fillRect(0, 0, 640, 360);
            
            // Draw visual loading scan lines
            ctx.strokeStyle = "#1e293b";
            ctx.lineWidth = 1;
            for (let i = 0; i < 360; i += 20) {
              ctx.beginPath();
              ctx.moveTo(0, i);
              ctx.lineTo(640, i);
              ctx.stroke();
            }

            ctx.fillStyle = "#38bdf8";
            ctx.shadowColor = "#38bdf8";
            ctx.shadowBlur = 8;
            ctx.font = "bold 16px Inter";
            ctx.fillText("AWAITING CAMERA AUTHORIZATION...", 180, 160);
            ctx.shadowBlur = 0;

            ctx.fillStyle = "#94a3b8";
            ctx.font = "12px JetBrains Mono";
            ctx.fillText("Provide camera access via prompt to test real-time browser tracking.", 100, 200);
          }
        } else {
          // Preset simulated detection & tracking pipelines
          processSimulatedTracking(ctx);
        }

        // Draw general high-tech HUD overlays
        if (showHudOverlay) {
          drawHudOverlay(ctx);
        }

        // GPU / FPS history metrics logger (throttled)
        if (stateRef.current.frameCount % 15 === 0) {
          const currentFps = Math.min(60, Math.floor(1000 / (elapsed || 1)));
          setFps(currentFps);
          
          setFpsHistory(prev => {
            const copy = [...prev.slice(1)];
            copy.push(currentFps);
            return copy;
          });

          setTrackHistoryCount(prev => {
            const copy = [...prev.slice(1)];
            copy.push(stateRef.current.objects.filter(o => o.status === "Active").length);
            return copy;
          });

          // Simulate GPU calculation loads depending on model size and confidence
          const baseGpuLoad = streamSource === "webcam" ? 55 : streamSource === "traffic" ? 42 : 32;
          setGpuLoad(Math.min(99, Math.floor(baseGpuLoad + (1 - confThreshold) * 15 + Math.random() * 6)));
        }

        // Mirror trackers to react state for rendering the telemetry tables
        if (stateRef.current.frameCount % 5 === 0) {
          setActiveTrackersList([...stateRef.current.objects]);
        }

        stateRef.current.lastTime = now;
      }

      animationId = requestAnimationFrame(renderLoop);
    };

    // Begin render loop
    animationId = requestAnimationFrame(renderLoop);

    return () => {
      cancelAnimationFrame(animationId);
    };
  }, [streamSource, webcamActive, useSimulatedWebcam, isPaused, showBoxes, showIds, showTrails, showKalmanPredict, showHudOverlay, showGrid, confThreshold, maxAge, minHits, targetFps, metricMode, uploadedMediaUrl, uploadedMediaType, isUploadedActive]);

  // Visual backgrounds
  const drawSimulatedOfficeCamera = (ctx: CanvasRenderingContext2D) => {
    const width = 640;
    const height = 360;
    
    // Background - Cozy Dark Tech Office Interior
    ctx.fillStyle = "#070c17";
    ctx.fillRect(0, 0, width, height);

    // Grid pattern for a cybernetic security scan feeling
    ctx.strokeStyle = "rgba(56, 189, 248, 0.05)";
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw some stylized office elements (desks, plants, windows)
    // Office partition/wall lines
    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, 120); ctx.lineTo(160, 120); ctx.lineTo(200, 360);
    ctx.moveTo(640, 120); ctx.lineTo(480, 120); ctx.lineTo(440, 360);
    ctx.stroke();

    // Stylized Desk at the center-bottom
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(200, 240, 240, 120);
    ctx.strokeStyle = "#334155";
    ctx.strokeRect(200, 240, 240, 120);

    // Laptop on the desk (Static laptop silhouette)
    ctx.fillStyle = "#1e293b";
    ctx.fillRect(280, 215, 80, 15); // screen
    ctx.fillRect(270, 230, 100, 10); // keyboard base
    ctx.fillStyle = "#38bdf8";
    ctx.globalAlpha = 0.15;
    ctx.fillRect(282, 217, 76, 11); // screen glow
    ctx.globalAlpha = 1.0;

    // A tech server rack on the left side
    ctx.fillStyle = "#020617";
    ctx.fillRect(20, 140, 100, 180);
    ctx.strokeStyle = "#1e293b";
    ctx.strokeRect(20, 140, 100, 180);
    
    // LED blinkers on server rack
    const t = stateRef.current.frameCount;
    for (let i = 0; i < 6; i++) {
      ctx.fillStyle = (t + i * 15) % 60 < 25 ? "#06b6d4" : "#1e293b"; // Cyan LEDs
      ctx.fillRect(35, 160 + i * 25, 8, 4);
      ctx.fillStyle = (t + i * 25) % 45 < 15 ? "#10b981" : "#1e293b"; // Emerald LEDs
      ctx.fillRect(50, 160 + i * 25, 8, 4);
    }

    // Now, let's draw some simulated interactive subjects that move and will trigger standard frame-differencing motion tracking!
    // Subject 1: A simulated Person (represented as a stylized glowing circular avatar or silhouette) walking across the office.
    // Standard frequency sinusoidal walking motion
    const personTime = t * 0.015;
    const px = 320 + Math.sin(personTime) * 160;
    const py = 160 + Math.cos(personTime * 2.3) * 20;

    // Draw "Person 1" head & body (slate-blue colored circle and round rect)
    ctx.fillStyle = "#475569";
    // Body
    ctx.beginPath();
    ctx.arc(px, py - 15, 15, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillRect(px - 18, py, 36, 50);
    // Dynamic hand-waving or head bobbing details to trigger rich motion vectors!
    ctx.strokeStyle = "#64748b";
    ctx.lineWidth = 6;
    ctx.lineCap = "round";
    // Left arm
    ctx.beginPath();
    ctx.moveTo(px - 18, py + 10);
    ctx.lineTo(px - 32, py + 15 + Math.sin(t * 0.08) * 15);
    ctx.stroke();
    // Right arm
    ctx.beginPath();
    ctx.moveTo(px + 18, py + 10);
    ctx.lineTo(px + 32, py + 15 + Math.cos(t * 0.08) * 15);
    ctx.stroke();

    // Subject 2: A small drone or security robot flying or rolling in the background
    const droneTime = t * 0.03;
    const dx = 320 + Math.cos(droneTime) * 220;
    const dy = 90 + Math.sin(droneTime * 1.5) * 25;

    ctx.fillStyle = "#334155";
    ctx.fillRect(dx - 12, dy - 6, 24, 12);
    // Rotor blades flashing lines (dynamic changes trigger motion differencing!)
    ctx.strokeStyle = (t % 4 < 2) ? "#475569" : "#334155";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(dx - 22, dy - 6); ctx.lineTo(dx - 2, dy - 6);
    ctx.moveTo(dx + 2, dy - 6); ctx.lineTo(dx + 22, dy - 6);
    ctx.stroke();

    // Add red recording camera indicator blinking in top right corner
    ctx.fillStyle = (t % 30 < 15) ? "#ef4444" : "rgba(239, 68, 68, 0.2)";
    ctx.beginPath();
    ctx.arc(600, 40, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 10px JetBrains Mono";
    ctx.fillText("REC LIVE (SIM)", 500, 44);
  };

  const drawTrafficBackground = (ctx: CanvasRenderingContext2D) => {
    // Crossroads asphalt
    ctx.fillStyle = "#1e293b";
    ctx.fillRect(0, 110, 640, 160); // Horiz road
    ctx.fillRect(240, 0, 160, 360); // Vert road

    // Road markings (yellow line center)
    ctx.strokeStyle = "#eab308";
    ctx.lineWidth = 3;
    ctx.setLineDash([10, 15]);
    
    // Horizontal center
    ctx.beginPath();
    ctx.moveTo(0, 190);
    ctx.lineTo(640, 190);
    ctx.stroke();

    // Vertical center
    ctx.beginPath();
    ctx.moveTo(320, 0);
    ctx.lineTo(320, 360);
    ctx.stroke();

    ctx.setLineDash([]);

    // Solid side walk white lines
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, 110); ctx.lineTo(640, 110);
    ctx.moveTo(0, 270); ctx.lineTo(640, 270);
    ctx.moveTo(240, 0); ctx.lineTo(240, 360);
    ctx.moveTo(400, 0); ctx.lineTo(400, 360);
    ctx.stroke();

    // Draw scanning grid overlay
    if (showGrid) {
      drawGridLayer(ctx);
    }

    // Overhead transparent pedestrian bridge (Occasion Zone!)
    // When objects go between x=430 and x=510, we make them "Occluded" to test SORT predictions!
    ctx.fillStyle = "rgba(100, 116, 139, 0.4)";
    ctx.fillRect(440, 110, 80, 160);
    
    // Bridge truss structure lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(440, 110); ctx.lineTo(520, 270);
    ctx.moveTo(520, 110); ctx.lineTo(440, 270);
    ctx.stroke();

    ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
    ctx.font = "bold 9px JetBrains Mono";
    ctx.fillText("OCCLUSION BRIDGE", 442, 125);
    ctx.font = "8px JetBrains Mono";
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.fillText("Kalman Testing Zone", 442, 138);
  };

  const drawPedestrianBackground = (ctx: CanvasRenderingContext2D) => {
    // Beautiful paved stone block pattern
    ctx.fillStyle = "#111827";
    ctx.fillRect(0, 0, 640, 360);

    ctx.strokeStyle = "#1f2937";
    ctx.lineWidth = 2;
    for (let i = 0; i < 640; i += 40) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, 360);
      ctx.stroke();
    }
    for (let j = 0; j < 360; j += 40) {
      ctx.beginPath();
      ctx.moveTo(0, j);
      ctx.lineTo(640, j);
      ctx.stroke();
    }

    // Central round fountain center
    ctx.fillStyle = "#1f2937";
    ctx.beginPath();
    ctx.arc(320, 180, 50, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#4b5563";
    ctx.stroke();

    // Central water pool
    ctx.fillStyle = "#1e40af";
    ctx.beginPath();
    ctx.arc(320, 180, 42, 0, Math.PI * 2);
    ctx.fill();

    if (showGrid) {
      drawGridLayer(ctx);
    }

    // Overhead transparent umbrella tree blocking pedestrian views
    ctx.fillStyle = "rgba(22, 101, 52, 0.35)";
    ctx.beginPath();
    ctx.arc(140, 140, 65, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(74, 222, 128, 0.25)";
    ctx.stroke();

    ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
    ctx.font = "bold 9px JetBrains Mono";
    ctx.fillText("TREELINE OCCLUSION", 95, 120);
  };

  const drawConveyorBackground = (ctx: CanvasRenderingContext2D) => {
    // Manufacturing assembly floor
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, 640, 360);

    // Assembly lines
    ctx.fillStyle = "#334155";
    ctx.fillRect(0, 150, 640, 80);

    // Rollers on belt
    ctx.fillStyle = "#1e293b";
    for (let i = 20; i < 640; i += 60) {
      ctx.fillRect(i, 152, 35, 76);
      ctx.fillStyle = "#475569";
      ctx.fillRect(i + 15, 152, 5, 76);
      ctx.fillStyle = "#1e293b";
    }

    // Guard rails
    ctx.fillStyle = "#64748b";
    ctx.fillRect(0, 146, 640, 4);
    ctx.fillRect(0, 230, 640, 4);

    if (showGrid) {
      drawGridLayer(ctx);
    }

    // Camera Scan Line Flash Overlay (YOLO Camera inspect location)
    ctx.fillStyle = "rgba(56, 189, 248, 0.04)";
    ctx.fillRect(360, 0, 120, 360);

    // Light flash bar
    ctx.strokeStyle = "rgba(56, 189, 248, 0.6)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(360, 0); ctx.lineTo(360, 360);
    ctx.moveTo(480, 0); ctx.lineTo(480, 360);
    ctx.stroke();

    ctx.fillStyle = "rgba(56, 189, 248, 0.8)";
    ctx.font = "bold 9px JetBrains Mono";
    ctx.fillText("YOLO INSPECTOR ZONE", 367, 30);
  };

  const drawGridLayer = (ctx: CanvasRenderingContext2D) => {
    ctx.strokeStyle = "rgba(56, 189, 248, 0.08)";
    ctx.lineWidth = 1;
    for (let i = 0; i < 640; i += 30) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, 360);
      ctx.stroke();
    }
    for (let j = 0; j < 360; j += 30) {
      ctx.beginPath();
      ctx.moveTo(0, j);
      ctx.lineTo(640, j);
      ctx.stroke();
    }
  };

  const drawHudOverlay = (ctx: CanvasRenderingContext2D) => {
    // Upper-left corner high tech HUD box
    ctx.fillStyle = "rgba(11, 15, 25, 0.85)";
    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 1.5;
    ctx.fillRect(10, 10, 210, 95);
    ctx.strokeRect(10, 10, 210, 95);

    // Glowing green status bullet
    ctx.fillStyle = "#10b981";
    ctx.shadowColor = "#10b981";
    ctx.shadowBlur = 4;
    ctx.beginPath();
    ctx.arc(25, 26, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.fillStyle = "#f1f5f9";
    ctx.font = "bold 10px JetBrains Mono";
    ctx.fillText("YOLO & SORT PIPELINE", 37, 29);

    ctx.strokeStyle = "rgba(51, 65, 85, 0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(20, 38);
    ctx.lineTo(210, 38);
    ctx.stroke();

    // Data readouts
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px JetBrains Mono";
    ctx.fillText("INF. SPEED : ", 22, 54);
    ctx.fillStyle = "#38bdf8";
    ctx.fillText(`${(1000 / fps).toFixed(1)} ms`, 110, 54);

    ctx.fillStyle = "#94a3b8";
    ctx.fillText("ENGINE FPS : ", 22, 70);
    ctx.fillStyle = "#10b981";
    ctx.fillText(`${fps.toFixed(1)} FPS`, 110, 70);

    ctx.fillStyle = "#94a3b8";
    ctx.fillText("TRACKS ACT : ", 22, 86);
    ctx.fillStyle = "#eab308";
    ctx.fillText(`${stateRef.current.objects.filter(o => o.status === "Active").length} units`, 110, 86);
    
    // Tiny telemetry logo decoration in upper right
    ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
    ctx.strokeRect(600, 15, 25, 25);
    ctx.beginPath();
    ctx.arc(612.5, 27.5, 6, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = "#38bdf8";
    ctx.fillRect(611, 26, 3, 3);
  };

  // --- PROCESSING FOR SIMULATED DETECTIONS AND SORT MATCHING ---
  const processSimulatedTracking = (ctx: CanvasRenderingContext2D) => {
    const list = stateRef.current.objects;

    // Filter out items that have completed their paths
    const activeAndLost = list.filter(obj => {
      if (streamSource === "traffic") {
        return obj.x1 > -150 && obj.x1 < 750;
      } else if (streamSource === "pedestrian") {
        return obj.x1 > -100 && obj.x1 < 740 && obj.y1 > -100 && obj.y1 < 460;
      } else {
        // Conveyor belt repeat loop
        if (obj.x1 > 660) {
          // Recycle object
          obj.x1 = -80;
          obj.x2 = -40;
          obj.id = stateRef.current.nextId++;
          obj.confidence = 0.95 + Math.random() * 0.04;
          obj.history = [];
          obj.lifetime = 0;
          setTotalObjectsCount(prev => prev + 1);
        }
        return true;
      }
    });

    stateRef.current.objects = activeAndLost;

    // Run motion updates and match YOLO predictions
    stateRef.current.objects.forEach(obj => {
      obj.lifetime++;

      // Compute displacement vector
      const dx = Math.cos(obj.direction) * obj.speed;
      const dy = Math.sin(obj.direction) * obj.speed;

      // Update true position of object
      obj.x1 += dx;
      obj.x2 += dx;
      obj.y1 += dy;
      obj.y2 += dy;

      // 1. Occlusion Simulation (SORT test)
      let inOcclusionZone = false;
      if (streamSource === "traffic" && obj.x1 > 425 && obj.x2 < 525) {
        inOcclusionZone = true;
      } else if (streamSource === "pedestrian" && Math.pow(obj.x1 - 130, 2) + Math.pow(obj.y1 - 130, 2) < Math.pow(55, 2)) {
        inOcclusionZone = true;
      }

      // Simulate confidence score drops
      // YOLO Confidence slider effect: If YOLO slider is higher than object confidence, we temporarily "lose" detection!
      const failedYoloConfCheck = obj.confidence < confThreshold;

      if (inOcclusionZone || failedYoloConfCheck) {
        obj.framesLost++;
        if (obj.framesLost < maxAge) {
          obj.status = "Lost"; // Enter Kalman state prediction mode
        } else {
          // Exceeded max age threshold: prune track
          obj.status = "Lost";
        }
      } else {
        obj.status = "Active";
        obj.framesLost = 0;
      }

      // Record trajectory history for trail rendering
      const cx = (obj.x1 + obj.x2) / 2;
      const cy = (obj.y1 + obj.y2) / 2;
      
      if (obj.status === "Active") {
        obj.history.push({ x: cx, y: cy });
        if (obj.history.length > 20) {
          obj.history.shift();
        }
      }

      // RENDER: Trails
      if (showTrails && obj.history.length > 1) {
        ctx.beginPath();
        ctx.strokeStyle = obj.color;
        ctx.lineWidth = 2.5;
        ctx.globalAlpha = obj.status === "Active" ? 0.75 : 0.3;
        ctx.moveTo(obj.history[0].x, obj.history[0].y);
        for (let idx = 1; idx < obj.history.length; idx++) {
          ctx.lineTo(obj.history[idx].x, obj.history[idx].y);
        }
        ctx.stroke();
        ctx.globalAlpha = 1.0;

        // Draw dot at tracker head
        ctx.fillStyle = obj.color;
        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, Math.PI * 2);
        ctx.fill();
      }

      // RENDER: Simulated bounding boxes & labels
      const width = obj.x2 - obj.x1;
      const height = obj.y2 - obj.y1;

      // Draw Kalman prediction state box if requested
      if (showKalmanPredict && obj.status === "Lost" && obj.framesLost < maxAge) {
        ctx.strokeStyle = "#eab308"; // Amber for Kalman prediction
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(obj.x1, obj.y1, width, height);
        ctx.setLineDash([]);

        // Label prediction
        ctx.fillStyle = "#eab308";
        ctx.font = "8px JetBrains Mono";
        ctx.fillText(`KALMAN EST ID ${obj.id}`, obj.x1 + 2, obj.y1 - 4);
      }

      // Draw normal box
      if (showBoxes && obj.status === "Active") {
        ctx.strokeStyle = obj.color;
        ctx.lineWidth = 2;
        ctx.strokeRect(obj.x1, obj.y1, width, height);

        // Draw high contrast visual corner brackets
        drawCornerBrackets(ctx, obj.x1, obj.y1, width, height, obj.color);

        // Label ID and Class
        if (showIds) {
          const badgeText = `ID ${obj.id} | ${obj.className} (${Math.floor(obj.confidence * 100)}%)`;
          ctx.font = "bold 9px JetBrains Mono";
          const measure = ctx.measureText(badgeText);
          const bgWidth = measure.width + 10;

          // Fill label background
          ctx.fillStyle = obj.color;
          ctx.fillRect(obj.x1 - 1, obj.y1 - 15, bgWidth, 15);

          // Draw text
          ctx.fillStyle = "#ffffff";
          ctx.fillText(badgeText, obj.x1 + 4, obj.y1 - 4);
        }
      }
    });

    // Handle conveyor conveyor analytics updates
    if (streamSource === "conveyor") {
      stateRef.current.objects.forEach(obj => {
        // Increment inventory counter once when it crosses scanner midline at x=420
        if (obj.x1 > 420 && obj.x1 < 425 && obj.lifetime > 0) {
          setClassCounts(prev => ({
            ...prev,
            "Object": prev["Object"] + 1
          }));
          setTotalObjectsCount(prev => prev + 1);
        }
      });
    }
  };

  // --- DRAWING SPECIAL CORNER BRACKETS FOR DETECTIONS ---
  const drawCornerBrackets = (ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, color: string) => {
    const len = Math.min(10, w / 4, h / 4);
    if (len < 2) return;

    ctx.strokeStyle = color;
    ctx.lineWidth = 3.5;

    ctx.beginPath();
    // Top-left
    ctx.moveTo(x + len, y); ctx.lineTo(x, y); ctx.lineTo(x, y + len);
    // Top-right
    ctx.moveTo(x + w - len, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + len);
    // Bottom-left
    ctx.moveTo(x, y + h - len); ctx.lineTo(x, y + h); ctx.lineTo(x + len, y + h);
    // Bottom-right
    ctx.moveTo(x + w - len, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - len);
    ctx.stroke();
  };

  // --- PIPELINE COMPONENT: ACTUAL REAL-TIME WEBCAM MOTION DETECTION AND SORT TRACKING ---
  const processWebcamTracking = (ctx: CanvasRenderingContext2D) => {
    // We downsample current frame and perform absolute frame differencing!
    // This provides standard "detections" that are associated and tracked.
    
    const width = 640;
    const height = 360;
    const sampleW = 120;
    const sampleH = 80;

    // 1. Grab pixel data of current frame
    let currentFrameData: ImageData;
    try {
      currentFrameData = ctx.getImageData(0, 0, width, height);
    } catch (e) {
      return;
    }

    // Create a temporary downsampled grey scale array (faster math)
    const gray = new Uint8Array(sampleW * sampleH);
    const stepX = width / sampleW;
    const stepY = height / sampleH;

    for (let y = 0; y < sampleH; y++) {
      for (let x = 0; x < sampleW; x++) {
        const srcIdx = Math.floor(y * stepY * width + x * stepX) * 4;
        const r = currentFrameData.data[srcIdx];
        const g = currentFrameData.data[srcIdx + 1];
        const b = currentFrameData.data[srcIdx + 2];
        // Standard luminance formula
        gray[y * sampleW + x] = 0.299 * r + 0.587 * g + 0.114 * b;
      }
    }

    // 2. Perform frame-differencing with previous frame
    const prevFrameData = stateRef.current.prevFrameData;
    const motionPixels: { x: number; y: number }[] = [];

    if (prevFrameData) {
      const threshold = 22; // Sensibility
      for (let i = 0; i < sampleW * sampleH; i++) {
        const diff = Math.abs(gray[i] - prevFrameData.data[i]);
        if (diff > threshold) {
          const y = Math.floor(i / sampleW);
          const x = i % sampleW;
          motionPixels.push({ x: x * stepX, y: y * stepY });
        }
      }
    }

    // Store current downsampled state for next tick frame difference
    // Construct fake ImageData wrapper for storing Uint8Array efficiently
    stateRef.current.prevFrameData = {
      data: gray as unknown as Uint8ClampedArray,
      width: sampleW,
      height: sampleH,
      colorSpace: "srgb"
    };

    // 3. Cluster motion pixels into raw bounding boxes (Detection Stage)
    // Simple greedy density box builder
    const detectionsList: { x1: number; y1: number; x2: number; y2: number; size: number }[] = [];
    const minPixelDensity = 12; // Minimum motion particles to warrant a bbox

    if (motionPixels.length > minPixelDensity) {
      // Find bounding box enclosing the overall major motion cluster
      let minX = width, maxX = 0, minY = height, maxY = 0;
      motionPixels.forEach(p => {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
      });

      const clusterW = maxX - minX;
      const clusterH = maxY - minY;

      // Add a bounding box if it matches standard human sizing profiles
      if (clusterW > 35 && clusterH > 35) {
        // Enlarge slightly for buffer padding
        detectionsList.push({
          x1: Math.max(0, minX - 10),
          y1: Math.max(0, minY - 10),
          x2: Math.min(width, maxX + 10),
          y2: Math.min(height, maxY + 10),
          size: motionPixels.length
        });
      }

      // Add supplementary smaller cluster trackers for hands/objects
      if (motionPixels.length > 60) {
        // Divide frame left/right and extract secondary boxes if active
        const midX = (minX + maxX) / 2;
        let leftMinX = width, leftMaxX = 0, leftMinY = height, leftMaxY = 0;
        let rightMinX = width, rightMaxX = 0, rightMinY = height, rightMaxY = 0;
        let leftCount = 0, rightCount = 0;

        motionPixels.forEach(p => {
          if (p.x < midX) {
            leftCount++;
            if (p.x < leftMinX) leftMinX = p.x;
            if (p.x > leftMaxX) leftMaxX = p.x;
            if (p.y < leftMinY) leftMinY = p.y;
            if (p.y > leftMaxY) leftMaxY = p.y;
          } else {
            rightCount++;
            if (p.x < rightMinX) rightMinX = p.x;
            if (p.x > rightMaxX) rightMaxX = p.x;
            if (p.y < rightMinY) rightMinY = p.y;
            if (p.y > rightMaxY) rightMaxY = p.y;
          }
        });

        if (leftCount > minPixelDensity && (leftMaxX - leftMinX) > 25) {
          detectionsList.push({
            x1: leftMinX, y1: leftMinY, x2: leftMaxX, y2: leftMaxY, size: leftCount
          });
        }
        if (rightCount > minPixelDensity && (rightMaxX - rightMinX) > 25) {
          detectionsList.push({
            x1: rightMinX, y1: rightMinY, x2: rightMaxX, y2: rightMaxY, size: rightCount
          });
        }
      }
    }

    // Inject simulated image detections for static uploaded images
    if (uploadedMediaType === "image") {
      const t = stateRef.current.frameCount;
      // Object 1: "Person" slowly walking/hovering
      const px = 220 + Math.sin(t * 0.02) * 90;
      const py = 170 + Math.cos(t * 0.01) * 20;
      detectionsList.push({
        x1: px - 25,
        y1: py - 55,
        x2: px + 25,
        y2: py + 55,
        size: 200
      });
      
      // Object 2: "Car" slowly moving
      const cx = 410 + Math.cos(t * 0.015) * 110;
      const cy = 230 + Math.sin(t * 0.008) * 10;
      detectionsList.push({
        x1: cx - 40,
        y1: cy - 20,
        x2: cx + 40,
        y2: cy + 20,
        size: 250
      });
    }

    // 4. TS SORT DATA ASSOCIATION ENGINE
    // We update active tracking lists using the new detections, matching via IoU overlap thresholds!
    const activeObjects = stateRef.current.objects;

    // Filter dead trackers
    const persistentObjects = activeObjects.filter(obj => {
      if (obj.status === "Lost" && obj.framesLost >= maxAge) {
        return false; // delete track
      }
      return true;
    });

    const updatedObjects: SimulatedObject[] = [];

    // Greedy IoU matching solver
    const matchedDetectionsIndices = new Set<number>();
    
    persistentObjects.forEach(trk => {
      trk.lifetime++;

      // Compute IoU overlap with all current frame detections
      let bestMatchIdx = -1;
      let maxIoU = 0;

      detectionsList.forEach((det, dIdx) => {
        if (matchedDetectionsIndices.has(dIdx)) return;

        // Compute IoU intersection
        const ix1 = Math.max(trk.x1, det.x1);
        const iy1 = Math.max(trk.y1, det.y1);
        const ix2 = Math.min(trk.x2, det.x2);
        const iy2 = Math.min(trk.y2, det.y2);

        const iw = Math.max(0, ix2 - ix1);
        const ih = Math.max(0, iy2 - iy1);
        const intersection = iw * ih;

        const area1 = (trk.x2 - trk.x1) * (trk.y2 - trk.y1);
        const area2 = (det.x2 - det.x1) * (det.y2 - det.y1);
        const union = area1 + area2 - intersection;

        const iou = union > 0 ? intersection / union : 0;
        if (iou > maxIoU) {
          maxIoU = iou;
          bestMatchIdx = dIdx;
        }
      });

      // Threshold check for association
      const iouThreshold = 0.22;
      if (bestMatchIdx !== -1 && maxIoU >= iouThreshold) {
        // Tracker successfully associated to new detection!
        const det = detectionsList[bestMatchIdx];
        matchedDetectionsIndices.add(bestMatchIdx);

        // State update with smooth alpha easing (simulating Kalman gain)
        const alpha = 0.55; 
        trk.x1 = trk.x1 + (det.x1 - trk.x1) * alpha;
        trk.y1 = trk.y1 + (det.y1 - trk.y1) * alpha;
        trk.x2 = trk.x2 + (det.x2 - trk.x2) * alpha;
        trk.y2 = trk.y2 + (det.y2 - trk.y2) * alpha;

        trk.status = "Active";
        trk.framesLost = 0;
        trk.confidence = Math.min(0.99, 0.70 + (det.size / 400));
        
        const cx = (trk.x1 + trk.x2) / 2;
        const cy = (trk.y1 + trk.y2) / 2;
        trk.history.push({ x: cx, y: cy });
        if (trk.history.length > 20) trk.history.shift();

        updatedObjects.push(trk);
      } else {
        // No match found: Object goes into Lost/Prediction mode
        trk.framesLost++;
        trk.status = "Lost";

        // Simulating Kalman displacement prediction based on history velocity
        if (trk.history.length >= 2) {
          const pt1 = trk.history[trk.history.length - 2];
          const pt2 = trk.history[trk.history.length - 1];
          const vx = pt2.x - pt1.x;
          const vy = pt2.y - pt1.y;

          // Propagate bounding box center
          trk.x1 += vx;
          trk.x2 += vx;
          trk.y1 += vy;
          trk.y2 += vy;
        }

        updatedObjects.push(trk);
      }
    });

    // 5. Create new Trackers for unmatched detections
    detectionsList.forEach((det, dIdx) => {
      if (matchedDetectionsIndices.has(dIdx)) return;

      // Ensure minimum scale density check to prevent noise initiation
      if (det.size > minPixelDensity * 2.5) {
        const id = stateRef.current.webcamTrackerId++;
        const width = det.x2 - det.x1;
        const aspect = width / (det.y2 - det.y1);
        
        // Dynamic labels based on size/aspect ratio
        let label = "Object";
        if (width > 220) label = "Person";
        else if (aspect > 1.2) label = "Hand";
        else if (width > 80 && width <= 220) label = "Face";

        const color = CLASS_COLORS[label] || "#38bdf8";

        // Add tracker
        updatedObjects.push({
          id: id,
          classId: label === "Person" ? 0 : label === "Face" ? 12 : 39,
          className: label,
          x1: det.x1,
          y1: det.y1,
          x2: det.x2,
          y2: det.y2,
          targetX: det.x1,
          targetY: det.y1,
          speed: 1,
          direction: 0,
          color: color,
          history: [{ x: (det.x1 + det.x2) / 2, y: (det.y1 + det.y2) / 2 }],
          status: "Active",
          framesLost: 0,
          confidence: Math.min(0.98, 0.72 + (det.size / 500)),
          lifetime: 1
        });

        // Increment stats counts
        setTotalObjectsCount(prev => prev + 1);
        setClassCounts(prev => {
          const cLabel = label === "Face" ? "Bicycle" : label === "Hand" ? "Truck" : label; // Map to base counts categories for simpler bar chart
          return {
            ...prev,
            [cLabel]: (prev[cLabel] || 0) + 1
          };
        });
      }
    });

    stateRef.current.objects = updatedObjects;

    // 6. DRAW Overlays on top of webcam stream
    stateRef.current.objects.forEach(obj => {
      const w = obj.x2 - obj.x1;
      const h = obj.y2 - obj.y1;

      // Historical trails
      if (showTrails && obj.history.length > 1) {
        ctx.beginPath();
        ctx.strokeStyle = obj.color;
        ctx.lineWidth = 3;
        ctx.globalAlpha = obj.status === "Active" ? 0.8 : 0.35;
        ctx.moveTo(obj.history[0].x, obj.history[0].y);
        for (let i = 1; i < obj.history.length; i++) {
          ctx.lineTo(obj.history[i].x, obj.history[i].y);
        }
        ctx.stroke();
        ctx.globalAlpha = 1.0;
      }

      // Predicted state box
      if (showKalmanPredict && obj.status === "Lost") {
        ctx.strokeStyle = "#eab308";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(obj.x1, obj.y1, w, h);
        ctx.setLineDash([]);

        ctx.fillStyle = "#eab308";
        ctx.font = "bold 8px JetBrains Mono";
        ctx.fillText(`KALMAN PREDICT ID ${obj.id}`, obj.x1 + 4, obj.y1 - 4);
      }

      // Detection Box
      if (showBoxes && obj.status === "Active") {
        ctx.strokeStyle = obj.color;
        ctx.lineWidth = 2.5;
        ctx.strokeRect(obj.x1, obj.y1, w, h);

        drawCornerBrackets(ctx, obj.x1, obj.y1, w, h, obj.color);

        if (showIds) {
          const labelText = `TRK #${obj.id} | ${obj.className}`;
          ctx.font = "bold 9px JetBrains Mono";
          const measure = ctx.measureText(labelText);

          // Fill background tag
          ctx.fillStyle = obj.color;
          ctx.fillRect(obj.x1 - 1, obj.y1 - 15, measure.width + 10, 15);

          ctx.fillStyle = "#ffffff";
          ctx.fillText(labelText, obj.x1 + 4, obj.y1 - 4);
        }
      }
    });
  };

  return (
    <div className="min-h-screen bg-[#070a13] text-slate-100 flex flex-col selection:bg-cyan-500 selection:text-slate-900" id="main_wrapper">
      
      {/* Sleek Navigation Header */}
      <header className="border-b border-slate-800 bg-[#0b0f19]/80 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex items-center justify-between" id="app_header">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-gradient-to-tr from-cyan-500 to-indigo-600 rounded-xl shadow-lg shadow-cyan-500/10 flex items-center justify-center">
            <Cpu className="w-5 h-5 text-slate-900 stroke-[2.5]" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
              YOLOv8 & Deep SORT
              <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                PRO PIPELINE
              </span>
            </h1>
            <p className="text-xs text-slate-400 font-medium">Real-Time Object Detection, Multi-Target Assignment & Tracking</p>
          </div>
        </div>

        {/* Tab Selection */}
        <div className="flex bg-slate-900/90 border border-slate-800 p-1 rounded-xl">
          <button
            onClick={() => setActiveTab("dashboard")}
            className={`px-4 py-2 text-xs font-semibold rounded-lg flex items-center gap-2 transition-all ${
              activeTab === "dashboard"
                ? "bg-slate-800 text-cyan-400 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            Live Dashboard
          </button>
          <button
            onClick={() => setActiveTab("code")}
            className={`px-4 py-2 text-xs font-semibold rounded-lg flex items-center gap-2 transition-all ${
              activeTab === "code"
                ? "bg-slate-800 text-cyan-400 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <FileCode className="w-3.5 h-3.5" />
            Python Source Repository
          </button>
          <button
            onClick={() => setActiveTab("guide")}
            className={`px-4 py-2 text-xs font-semibold rounded-lg flex items-center gap-2 transition-all ${
              activeTab === "guide"
                ? "bg-slate-800 text-cyan-400 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            Engineering Guide
          </button>
        </div>
      </header>

      {/* Main Content Areas */}
      <main className="flex-1 p-6 max-w-7xl w-full mx-auto" id="app_content">
        <AnimatePresence mode="wait">
          
          {/* TAB 1: VISUAL SIMULATOR & DASHBOARD */}
          {activeTab === "dashboard" && (
            <motion.div
              key="dashboard-view"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="grid grid-cols-1 lg:grid-cols-12 gap-6"
            >
              
              {/* LEFT VIEW: Interactive Simulator Screen & Controller */}
              <div className="lg:col-span-8 flex flex-col gap-6">
                
                {/* Direct Webcam Access Alert Call-To-Action */}
                {streamSource !== "webcam" && (
                  <motion.div 
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    className="bg-[#0b1120] border border-cyan-500/20 rounded-2xl p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-xl"
                  >
                    <div className="flex items-center gap-3">
                      <div className="p-2.5 bg-cyan-500/10 rounded-xl border border-cyan-500/20 flex items-center justify-center text-cyan-400">
                        <Video className="w-5 h-5 animate-pulse" />
                      </div>
                      <div>
                        <h4 className="text-xs font-bold text-white tracking-wide uppercase">Try Live Webcam Object Tracking!</h4>
                        <p className="text-[11px] text-slate-400 mt-0.5">
                          Perform real-time motion detection and Kalman-SORT multi-target tracking on your own camera stream.
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={() => setStreamSource("webcam")}
                      className="px-4 py-2 bg-gradient-to-r from-cyan-500 to-indigo-600 text-slate-950 hover:from-cyan-400 hover:to-indigo-500 font-bold rounded-xl text-xs flex items-center justify-center gap-1.5 transition shadow-lg shadow-cyan-500/10 self-start sm:self-auto"
                    >
                      <Video className="w-3.5 h-3.5 text-slate-950 stroke-[2.5]" />
                      Launch Webcam
                    </button>
                  </motion.div>
                )}

                {/* Visualizer Canvas Card */}
                <div className="bg-[#0b1120] border border-slate-800/80 rounded-2xl overflow-hidden shadow-2xl relative">
                  
                  {/* Status Banner */}
                  <div className="bg-[#0f172a] px-5 py-3 flex items-center justify-between border-b border-slate-800/60">
                    <div className="flex items-center gap-3">
                      <div className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
                      <span className="text-xs font-mono font-semibold tracking-wider text-slate-300 uppercase">
                        Stream Source: {isUploadedActive ? `Imported File: ${uploadedMediaName}` : streamSource === "webcam" ? "Real-time Web Camera Capture" : `${streamSource} simulator`}
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      {isUploadedActive && (
                        <button
                          onClick={clearUploadedMedia}
                          className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-950/40 hover:bg-rose-900/50 text-rose-400 border border-rose-800/40 transition flex items-center gap-1"
                        >
                          ✕ Clear File
                        </button>
                      )}
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-cyan-400 border border-slate-700/60 font-bold">
                        {isUploadedActive ? `File Track (${uploadedMediaType})` : streamSource === "webcam" ? "Real-time OpenCV Diff" : "YOLOv8 + SORT"}
                      </span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-emerald-400 border border-slate-700/60 font-bold">
                        GPU Active (CUDA)
                      </span>
                    </div>
                  </div>

                  {/* Hidden webcam/uploaded video tag to capture stream (opacity-0 instead of hidden to prevent browser decoding/play freeze) */}
                  <video 
                    ref={videoRef} 
                    className="absolute pointer-events-none opacity-0 w-1 h-1" 
                    playsInline 
                    muted 
                    autoPlay
                    width="640" 
                    height="360"
                  />

                  {/* Hidden image tag for uploaded static photos */}
                  {uploadedMediaType === "image" && uploadedMediaUrl && (
                    <img 
                      ref={imgRef}
                      src={uploadedMediaUrl}
                      className="absolute pointer-events-none opacity-0 w-1 h-1" 
                      alt="Uploaded media source"
                      referrerPolicy="no-referrer"
                    />
                  )}

                  {/* Canvas Visualizer */}
                  <div className="aspect-video w-full bg-[#030712] relative flex items-center justify-center">
                    <canvas 
                      ref={canvasRef} 
                      width="640" 
                      height="360" 
                      className="w-full h-full object-contain"
                    />

                    {/* Webcam Activation Overlay */}
                    {streamSource === "webcam" && !webcamActive && !useSimulatedWebcam && (
                      <div className="absolute inset-0 bg-[#070a13]/95 flex flex-col items-center justify-center p-6 text-center z-20 overflow-y-auto">
                        <div className="w-14 h-14 rounded-full bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center mb-3 relative">
                          <Video className="w-7 h-7 text-cyan-400" />
                          <div className="absolute inset-0 rounded-full border border-cyan-400 animate-ping opacity-25" />
                        </div>
                        
                        <h3 className="text-sm font-bold text-white tracking-wider uppercase mb-1">
                          Activate Live Webcam Tracking
                        </h3>
                        <p className="text-xs text-slate-400 max-w-md mb-4 leading-relaxed">
                          This uses your browser's webcam to perform real-time motion clustering and SORT track association locally in your browser.
                        </p>

                        {webcamError ? (
                          <div className="mb-4 bg-rose-500/10 border border-rose-500/20 px-4 py-3 rounded-xl max-w-lg text-left">
                            <p className="text-[12px] text-rose-400 flex items-start gap-2 mb-2">
                              <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5 text-rose-400" />
                              <span>
                                <strong>Webcam Error:</strong> Camera access was denied or is blocked in this container/iframe.
                              </span>
                            </p>
                            
                            <div className="border-t border-slate-800/80 pt-2">
                              <h4 className="text-[11px] font-bold text-slate-200 tracking-wider uppercase mb-1">
                                How to allow camera permission:
                              </h4>
                              <ul className="space-y-1 text-[11px] text-slate-400">
                                <li className="flex items-start gap-1.5">
                                  <span className="text-cyan-400 font-bold">1.</span>
                                  <span>
                                    <strong>Check Site Permissions:</strong> Click the <strong>Lock (🔒) icon</strong> on the left side of your browser's address bar and set <strong>Camera</strong> to <strong>"Allow"</strong>.
                                  </span>
                                </li>
                                <li className="flex items-start gap-1.5">
                                  <span className="text-cyan-400 font-bold">2.</span>
                                  <span>
                                    <strong>Use Simulation Fallback:</strong> If your environment doesn't have a webcam, click the green button below to test the computer vision code using a simulated room!
                                  </span>
                                </li>
                              </ul>
                            </div>
                          </div>
                        ) : null}

                        <div className="flex flex-col sm:flex-row gap-2.5 justify-center w-full max-w-lg">
                          <button
                            onClick={() => {
                              setUseSimulatedWebcam(true);
                              setWebcamError(null);
                            }}
                            className="px-4 py-2 bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-slate-950 font-extrabold rounded-xl text-xs flex items-center justify-center gap-2 transition shadow-lg shadow-emerald-500/20"
                          >
                            <Play className="w-3.5 h-3.5 text-slate-950 fill-slate-950" />
                            Use Simulated Camera
                          </button>

                          <button
                            onClick={startWebcam}
                            className="px-4 py-2 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold rounded-xl text-xs flex items-center justify-center gap-2 transition shadow-lg shadow-cyan-500/25"
                          >
                            <RefreshCw className="w-3.5 h-3.5 text-slate-950" />
                            {webcamError ? "Try Again" : "Grant Camera Permission"}
                          </button>

                          <a
                            href={window.location.href}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="px-4 py-2 bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-slate-300 font-bold rounded-xl text-xs flex items-center justify-center gap-2 transition"
                          >
                            <span>Open in Full Tab ↗</span>
                          </a>
                        </div>

                        <p className="text-[10px] text-slate-500 mt-3 max-w-xs leading-normal">
                          If your browser blocks camera access inside embed frames, click "Open in Full Tab ↗" or select "Use Simulated Camera".
                        </p>

                        <div className="mt-6 pt-5 border-t border-slate-800/80 w-full max-w-lg">
                          <p className="text-[10px] font-bold text-cyan-400 uppercase tracking-widest mb-3">
                            Or Detect Custom Media File
                          </p>
                          <label className="inline-flex items-center gap-2 px-5 py-2.5 bg-cyan-950/30 hover:bg-cyan-950/50 border border-cyan-800/40 text-cyan-400 font-extrabold rounded-xl text-xs cursor-pointer transition shadow-lg shadow-slate-950/40 hover:text-cyan-300">
                            <Upload className="w-4 h-4 text-cyan-400" />
                            <span>Import Video or Photo</span>
                            <input
                              type="file"
                              accept="video/*, image/*"
                              className="hidden"
                              onChange={handleFileUpload}
                            />
                          </label>
                          <p className="text-[9px] text-slate-500 mt-2">
                            Supports MP4, WebM, MOV videos or JPG, PNG images. Tracking runs locally.
                          </p>
                        </div>
                      </div>
                    )}

                    {/* GPU Load Overlay */}
                    <div className="absolute bottom-4 right-4 bg-slate-950/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-800 text-right font-mono">
                      <p className="text-[9px] text-slate-400 font-bold">GPU LOAD</p>
                      <p className={`text-xs font-bold ${gpuLoad > 75 ? "text-rose-400" : "text-cyan-400"}`}>
                        {gpuLoad}%
                      </p>
                    </div>

                    {/* Simulation Interrupted Overlay */}
                    {isPaused && (
                      <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm flex flex-col items-center justify-center gap-3">
                        <div className="p-4 bg-slate-900 border border-slate-700/60 rounded-full text-amber-400">
                          <Pause className="w-8 h-8 fill-amber-400/20" />
                        </div>
                        <p className="text-sm font-semibold text-white tracking-wide">DETECTION PIPELINE PAUSED</p>
                        <button
                          onClick={() => setIsPaused(false)}
                          className="px-4 py-1.5 bg-cyan-500 text-slate-950 text-xs font-bold rounded-lg hover:bg-cyan-400 transition"
                        >
                          Resume Tracking
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Stream Controls bar */}
                  <div className="bg-[#0f172a] p-4 flex flex-wrap items-center justify-between gap-4 border-t border-slate-800/60">
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => setIsPaused(!isPaused)}
                        className={`p-2 rounded-lg border transition ${
                          isPaused 
                            ? "bg-emerald-500 border-emerald-400 text-slate-950 hover:bg-emerald-400" 
                            : "bg-slate-800 border-slate-700 hover:bg-slate-700 text-slate-300"
                        }`}
                        title={isPaused ? "Play Stream" : "Pause Stream"}
                      >
                        {isPaused ? <Play className="w-4 h-4 fill-slate-950" /> : <Pause className="w-4 h-4" />}
                      </button>

                      <button
                        onClick={() => {
                          stateRef.current.objects = [];
                          stateRef.current.frameCount = 0;
                        }}
                        className="p-2 rounded-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 text-slate-300 transition animate-fade-in"
                        title="Reset Track IDs"
                      >
                        <RefreshCw className="w-4 h-4" />
                      </button>

                      {streamSource === "webcam" && (
                        <label className="p-2 rounded-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 hover:border-slate-600 text-slate-300 hover:text-cyan-400 transition cursor-pointer flex items-center justify-center gap-1.5 text-xs font-bold px-3 shadow-md">
                          <Upload className="w-3.5 h-3.5 text-cyan-400" />
                          <span>Import File</span>
                          <input
                            type="file"
                            accept="video/*, image/*"
                            className="hidden"
                            onChange={handleFileUpload}
                          />
                        </label>
                      )}
                    </div>

                    {/* Stream Selection Dropdown */}
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-400 font-semibold flex items-center gap-1">
                        <Video className="w-3.5 h-3.5 text-cyan-400" /> Source:
                      </span>
                      <div className="flex bg-slate-950 p-0.5 border border-slate-800 rounded-lg">
                        <button
                          onClick={() => setStreamSource("traffic")}
                          className={`px-3 py-1.5 text-[11px] font-semibold rounded-md transition ${
                            streamSource === "traffic" 
                              ? "bg-slate-800 text-cyan-400" 
                              : "text-slate-400 hover:text-slate-200"
                          }`}
                        >
                          Highway Traffic
                        </button>
                        <button
                          onClick={() => setStreamSource("pedestrian")}
                          className={`px-3 py-1.5 text-[11px] font-semibold rounded-md transition ${
                            streamSource === "pedestrian" 
                              ? "bg-slate-800 text-cyan-400" 
                              : "text-slate-400 hover:text-slate-200"
                          }`}
                        >
                          Pedestrian Crossing
                        </button>
                        <button
                          onClick={() => setStreamSource("conveyor")}
                          className={`px-3 py-1.5 text-[11px] font-semibold rounded-md transition ${
                            streamSource === "conveyor" 
                              ? "bg-slate-800 text-cyan-400" 
                              : "text-slate-400 hover:text-slate-200"
                          }`}
                        >
                          Conveyor Belt
                        </button>
                        <button
                          onClick={() => setStreamSource("webcam")}
                          className={`px-3 py-1.5 text-[11px] font-semibold rounded-md transition ${
                            streamSource === "webcam" 
                              ? "bg-slate-800 text-cyan-400" 
                              : "text-slate-400 hover:text-slate-200"
                          }`}
                        >
                          Live Webcam
                        </button>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Webcam Activation Warning Info Box */}
                {streamSource === "webcam" && webcamError && (
                  <div className="bg-rose-500/10 border border-rose-500/20 p-4 rounded-xl flex items-start gap-3">
                    <AlertCircle className="w-5 h-5 text-rose-400 flex-shrink-0" />
                    <div>
                      <h4 className="text-xs font-bold text-rose-300">Webcam Capture Error</h4>
                      <p className="text-[11px] text-rose-400 mt-1 leading-relaxed">{webcamError}</p>
                      <div className="flex gap-2 mt-2.5">
                        <button 
                          onClick={startWebcam}
                          className="text-[10px] text-white bg-slate-800 px-3 py-1.5 rounded-lg border border-slate-700 hover:bg-slate-700 transition"
                        >
                          Retry Connection
                        </button>
                        <button 
                          onClick={() => {
                            setUseSimulatedWebcam(true);
                            setWebcamError(null);
                          }}
                          className="text-[10px] text-slate-950 bg-gradient-to-r from-emerald-400 to-teal-500 hover:from-emerald-300 hover:to-teal-400 px-3 py-1.5 rounded-lg font-bold transition"
                        >
                          Use Simulated Camera Feed
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Control Panel: Parameters Sliders & Overlays togglers */}
                <div className="bg-[#0b1120] border border-slate-800/80 rounded-2xl p-6 shadow-xl grid grid-cols-1 md:grid-cols-2 gap-6">
                  
                  {/* Pipeline Parameters */}
                  <div className="flex flex-col gap-4">
                    <h3 className="text-xs font-bold tracking-wider text-slate-300 uppercase flex items-center gap-2">
                      <SlidersHorizontal className="w-4 h-4 text-cyan-400" />
                      YOLO & Tracker Hyperparameters
                    </h3>

                    {/* Confidence Slider */}
                    <div className="bg-slate-950/40 p-3.5 rounded-xl border border-slate-900/60">
                      <div className="flex justify-between text-xs mb-2">
                        <span className="text-slate-400 font-semibold">YOLO Confidence Threshold</span>
                        <span className="font-mono text-cyan-400 font-bold">{confThreshold.toFixed(2)}</span>
                      </div>
                      <input 
                        type="range" 
                        min="0.05" 
                        max="0.95" 
                        step="0.05" 
                        value={confThreshold}
                        onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                        className="w-full accent-cyan-500 bg-slate-800 rounded-lg cursor-pointer h-1.5"
                      />
                      <p className="text-[10px] text-slate-500 mt-2">
                        Lower shows more predictions but adds background noise. Higher filters weaker detections.
                      </p>
                    </div>

                    {/* Tracker Max Age Slider */}
                    <div className="bg-slate-950/40 p-3.5 rounded-xl border border-slate-900/60">
                      <div className="flex justify-between text-xs mb-2">
                        <span className="text-slate-400 font-semibold">Deep SORT Max Age (Frames)</span>
                        <span className="font-mono text-amber-400 font-bold">{maxAge} f</span>
                      </div>
                      <input 
                        type="range" 
                        min="2" 
                        max="45" 
                        step="1" 
                        value={maxAge}
                        onChange={(e) => setMaxAge(parseInt(e.target.value))}
                        className="w-full accent-amber-500 bg-slate-800 rounded-lg cursor-pointer h-1.5"
                      />
                      <p className="text-[10px] text-slate-500 mt-2">
                        Number of missing frames before the algorithm prunes a missing object track ID.
                      </p>
                    </div>

                    {/* Min Hits Slider */}
                    <div className="bg-slate-950/40 p-3.5 rounded-xl border border-slate-900/60">
                      <div className="flex justify-between text-xs mb-2">
                        <span className="text-slate-400 font-semibold">Min Init Hits (Frames)</span>
                        <span className="font-mono text-purple-400 font-bold">{minHits} f</span>
                      </div>
                      <input 
                        type="range" 
                        min="1" 
                        max="10" 
                        step="1" 
                        value={minHits}
                        onChange={(e) => setMinHits(parseInt(e.target.value))}
                        className="w-full accent-purple-500 bg-slate-800 rounded-lg cursor-pointer h-1.5"
                      />
                      <p className="text-[10px] text-slate-500 mt-2">
                        Consecutive detections needed to declare a track active and draw on the screen.
                      </p>
                    </div>
                  </div>

                  {/* UI Render Filters */}
                  <div className="flex flex-col gap-4">
                    <h3 className="text-xs font-bold tracking-wider text-slate-300 uppercase flex items-center gap-2">
                      <Layers className="w-4 h-4 text-cyan-400" />
                      Visual Analytics Overlay Toggles
                    </h3>

                    <div className="grid grid-cols-2 gap-3.5">
                      {/* Bounding Boxes */}
                      <button
                        onClick={() => setShowBoxes(!showBoxes)}
                        className={`p-3 rounded-xl border text-left flex items-center justify-between transition-all ${
                          showBoxes 
                            ? "bg-slate-900 border-cyan-500/30 text-white shadow-sm shadow-cyan-500/5" 
                            : "bg-slate-950/20 border-slate-900 text-slate-500"
                        }`}
                      >
                        <span className="text-xs font-semibold">Bounding Boxes</span>
                        {showBoxes ? <Eye className="w-3.5 h-3.5 text-cyan-400" /> : <EyeOff className="w-3.5 h-3.5" />}
                      </button>

                      {/* Track IDs */}
                      <button
                        onClick={() => setShowIds(!showIds)}
                        className={`p-3 rounded-xl border text-left flex items-center justify-between transition-all ${
                          showIds 
                            ? "bg-slate-900 border-cyan-500/30 text-white shadow-sm shadow-cyan-500/5" 
                            : "bg-slate-950/20 border-slate-900 text-slate-500"
                        }`}
                      >
                        <span className="text-xs font-semibold">Tracking Labels</span>
                        {showIds ? <Eye className="w-3.5 h-3.5 text-cyan-400" /> : <EyeOff className="w-3.5 h-3.5" />}
                      </button>

                      {/* Trailing history */}
                      <button
                        onClick={() => setShowTrails(!showTrails)}
                        className={`p-3 rounded-xl border text-left flex items-center justify-between transition-all ${
                          showTrails 
                            ? "bg-slate-900 border-cyan-500/30 text-white shadow-sm shadow-cyan-500/5" 
                            : "bg-slate-950/20 border-slate-900 text-slate-500"
                        }`}
                      >
                        <span className="text-xs font-semibold">Motion Trails</span>
                        {showTrails ? <Eye className="w-3.5 h-3.5 text-cyan-400" /> : <EyeOff className="w-3.5 h-3.5" />}
                      </button>

                      {/* Kalman State Box */}
                      <button
                        onClick={() => setShowKalmanPredict(!showKalmanPredict)}
                        className={`p-3 rounded-xl border text-left flex items-center justify-between transition-all ${
                          showKalmanPredict 
                            ? "bg-slate-900 border-cyan-500/30 text-white shadow-sm shadow-cyan-500/5" 
                            : "bg-slate-950/20 border-slate-900 text-slate-500"
                        }`}
                      >
                        <span className="text-xs font-semibold">Kalman States</span>
                        {showKalmanPredict ? <Eye className="w-3.5 h-3.5 text-cyan-400" /> : <EyeOff className="w-3.5 h-3.5" />}
                      </button>

                      {/* HUD overlay */}
                      <button
                        onClick={() => setShowHudOverlay(!showHudOverlay)}
                        className={`p-3 rounded-xl border text-left flex items-center justify-between transition-all ${
                          showHudOverlay 
                            ? "bg-slate-900 border-cyan-500/30 text-white shadow-sm shadow-cyan-500/5" 
                            : "bg-slate-950/20 border-slate-900 text-slate-500"
                        }`}
                      >
                        <span className="text-xs font-semibold">HUD Statistics</span>
                        {showHudOverlay ? <Eye className="w-3.5 h-3.5 text-cyan-400" /> : <EyeOff className="w-3.5 h-3.5" />}
                      </button>

                      {/* Grid background */}
                      <button
                        onClick={() => setShowGrid(!showGrid)}
                        className={`p-3 rounded-xl border text-left flex items-center justify-between transition-all ${
                          showGrid 
                            ? "bg-slate-900 border-cyan-500/30 text-white shadow-sm shadow-cyan-500/5" 
                            : "bg-slate-950/20 border-slate-900 text-slate-500"
                        }`}
                      >
                        <span className="text-xs font-semibold">Technical Grid</span>
                        {showGrid ? <Eye className="w-3.5 h-3.5 text-cyan-400" /> : <EyeOff className="w-3.5 h-3.5" />}
                      </button>
                    </div>

                    {/* Metric Selection */}
                    <div className="bg-slate-950/40 p-4 rounded-xl border border-slate-900/60 mt-2">
                      <p className="text-xs text-slate-400 font-semibold mb-2">Distance Association Metric</p>
                      <div className="grid grid-cols-2 gap-2">
                        <button
                          onClick={() => setMetricMode("iou")}
                          className={`py-1.5 text-[10px] font-mono rounded border transition ${
                            metricMode === "iou" 
                              ? "bg-slate-800 text-cyan-400 border-cyan-500/30" 
                              : "text-slate-500 border-slate-900 hover:text-slate-300"
                          }`}
                        >
                          IoU Overlap Cost
                        </button>
                        <button
                          onClick={() => setMetricMode("cosine")}
                          className={`py-1.5 text-[10px] font-mono rounded border transition ${
                            metricMode === "cosine" 
                              ? "bg-slate-800 text-cyan-400 border-cyan-500/30" 
                              : "text-slate-500 border-slate-900 hover:text-slate-300"
                          }`}
                        >
                          Deep Features Cosine
                        </button>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Live Tracking Telemetry Table */}
                <div className="bg-[#0b1120] border border-slate-800/80 rounded-2xl overflow-hidden shadow-xl">
                  <div className="px-6 py-4 border-b border-slate-800/60 bg-[#0f172a] flex items-center justify-between">
                    <h3 className="text-xs font-bold tracking-wider text-slate-300 uppercase flex items-center gap-2">
                      <Terminal className="w-4 h-4 text-cyan-400" />
                      Active Trackers Telemetry Log
                    </h3>
                    <span className="text-[10px] font-mono bg-slate-950 px-2.5 py-0.5 rounded text-cyan-400">
                      {activeTrackersList.length} Active Tracks
                    </span>
                  </div>

                  <div className="overflow-x-auto max-h-[300px] overflow-y-auto">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="border-b border-slate-800 text-slate-400 font-semibold bg-slate-950/40">
                          <th className="px-6 py-3">TRACK ID</th>
                          <th className="px-6 py-3">CLASS</th>
                          <th className="px-6 py-3">COORDINATES (X, Y)</th>
                          <th className="px-6 py-3">CONFIDENCE</th>
                          <th className="px-6 py-3">STATUS</th>
                          <th className="px-6 py-3 text-right">LIFETIME (FRM)</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60">
                        {activeTrackersList.length === 0 ? (
                          <tr>
                            <td colSpan={6} className="text-center py-8 text-slate-500 font-mono">
                              No active trackers recorded. Start stream or adjust parameters.
                            </td>
                          </tr>
                        ) : (
                          activeTrackersList.map((obj) => (
                            <tr key={obj.id} className="hover:bg-slate-900/40 transition">
                              <td className="px-6 py-2.5 font-mono font-bold text-white">
                                <span className="inline-block w-2.5 h-2.5 rounded mr-2" style={{ backgroundColor: obj.color }} />
                                #{obj.id}
                              </td>
                              <td className="px-6 py-2.5">
                                <span 
                                  className="px-2 py-0.5 rounded text-[10px] font-semibold border"
                                  style={{ 
                                    backgroundColor: `${obj.color}15`, 
                                    borderColor: `${obj.color}35`,
                                    color: obj.color
                                  }}
                                >
                                  {obj.className}
                                </span>
                              </td>
                              <td className="px-6 py-2.5 font-mono text-slate-400">
                                {`[${Math.floor((obj.x1 + obj.x2)/2)}, ${Math.floor((obj.y1 + obj.y2)/2)}]`}
                              </td>
                              <td className="px-6 py-2.5 font-mono text-slate-400">
                                {(obj.confidence * 100).toFixed(0)}%
                              </td>
                              <td className="px-6 py-2.5">
                                <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-mono ${
                                  obj.status === "Active" 
                                    ? "bg-emerald-500/10 text-emerald-400" 
                                    : "bg-amber-500/10 text-amber-400 animate-pulse"
                                }`}>
                                  <span className={`w-1.5 h-1.5 rounded-full ${obj.status === "Active" ? "bg-emerald-400" : "bg-amber-400"}`} />
                                  {obj.status}
                                </span>
                              </td>
                              <td className="px-6 py-2.5 text-right font-mono text-slate-400">
                                {obj.lifetime}
                              </td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              {/* RIGHT VIEW: Numerical Analytics & Pipeline Overview */}
              <div className="lg:col-span-4 flex flex-col gap-6">
                
                {/* Statistics counts panel */}
                <div className="bg-[#0b1120] border border-slate-800/80 rounded-2xl p-6 shadow-xl flex flex-col gap-6">
                  <h3 className="text-xs font-bold tracking-wider text-slate-300 uppercase flex items-center gap-2">
                    <Activity className="w-4 h-4 text-cyan-400" />
                    Accumulated Track Analytics
                  </h3>

                  {/* Total objects banner */}
                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-900/80 text-center relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-3 opacity-10">
                      <Cpu className="w-16 h-16 text-cyan-400" />
                    </div>
                    <p className="text-[10px] text-slate-400 font-mono tracking-wider">TOTAL PROCESSED UNIQUE TARGETS</p>
                    <p className="text-4xl font-bold text-white mt-1 tracking-tight">{totalObjectsCount}</p>
                    <p className="text-[9px] text-emerald-400 mt-1 font-mono flex items-center justify-center gap-1">
                      <Sparkles className="w-3 h-3" /> Database synchronized in real time
                    </p>
                  </div>

                  {/* Class Distribution charts */}
                  <div className="flex flex-col gap-3">
                    <p className="text-xs font-semibold text-slate-400">Objects Profile Distribution</p>
                    
                    {Object.entries(classCounts)
                      .filter(([_, val]) => (val as number) > 0 || streamSource === "conveyor")
                      .map(([key, val]) => {
                        const total = Object.values(classCounts).reduce((a, b) => (a as number) + (b as number), 0) || 1;
                        const percentage = Math.floor(((val as number) / (total as number)) * 100);
                        const barColor = CLASS_COLORS[key] || "#10b981";

                        return (
                          <div key={key} className="bg-slate-950/30 p-3 rounded-lg border border-slate-900/60">
                            <div className="flex justify-between text-xs mb-1">
                              <span className="font-semibold text-slate-300">{key}s</span>
                              <span className="font-mono text-slate-400">{val} ({percentage}%)</span>
                            </div>
                            <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden">
                              <div 
                                className="h-full rounded-full transition-all duration-500" 
                                style={{ 
                                  width: `${percentage}%`,
                                  backgroundColor: barColor 
                                }}
                              />
                            </div>
                          </div>
                        );
                      })}
                  </div>

                  {/* Small telemetry graph for FPS */}
                  <div>
                    <p className="text-xs font-semibold text-slate-400 mb-2">FPS Pipeline Stability</p>
                    <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-900/80 aspect-[16/7] flex items-end gap-1 px-4">
                      {fpsHistory.map((val, idx) => {
                        const heightPercent = Math.min(100, Math.floor((val / 60) * 100));
                        return (
                          <div 
                            key={idx} 
                            className="flex-1 bg-cyan-500/20 rounded-t-sm transition-all relative group"
                            style={{ height: `${heightPercent}%` }}
                          >
                            {/* Hover tooltip */}
                            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block bg-slate-900 text-[8px] px-1 py-0.5 rounded font-mono text-white border border-slate-700 whitespace-nowrap z-10">
                              {val} FPS
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Active trackers timeline */}
                  <div>
                    <p className="text-xs font-semibold text-slate-400 mb-2">Active Multi-Track Density</p>
                    <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-900/80 aspect-[16/7] flex items-end gap-1 px-4">
                      {trackHistoryCount.map((val, idx) => {
                        const heightPercent = Math.min(100, Math.floor((val / 10) * 100));
                        return (
                          <div 
                            key={idx} 
                            className="flex-1 bg-amber-500/25 rounded-t-sm transition-all relative group"
                            style={{ height: `${heightPercent}%` }}
                          >
                            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block bg-slate-900 text-[8px] px-1 py-0.5 rounded font-mono text-white border border-slate-700 whitespace-nowrap z-10">
                              {val} Active
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* Quick Info Block */}
                <div className="bg-slate-900/60 border border-slate-800 p-6 rounded-2xl flex flex-col gap-3">
                  <h4 className="text-xs font-bold text-white flex items-center gap-1.5">
                    <HelpCircle className="w-4 h-4 text-cyan-400" />
                    How to Test Occlusions:
                  </h4>
                  <ul className="text-[11px] text-slate-400 space-y-2 list-disc list-inside">
                    <li>Select the <strong className="text-slate-300">Highway Traffic</strong> preset.</li>
                    <li>Observe vehicles moving horizontally from left to right.</li>
                    <li>When a vehicle enters the shaded <strong className="text-slate-300">Occlusion Bridge</strong>, its frame detection is hidden.</li>
                    <li>Notice that its state box turns <strong className="text-amber-400 font-mono">Dotted Yellow</strong>. This represents the Kalman Filter predicting its velocity and position mathematically.</li>
                    <li>When it emerges, the tracker successfully associates the bounding box, maintaining the <strong className="text-white">exact same unique Track ID</strong>.</li>
                  </ul>
                </div>
              </div>
            </motion.div>
          )}

          {/* TAB 2: PYTHON CODE HUB */}
          {activeTab === "code" && (
            <motion.div
              key="code-view"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="grid grid-cols-1 lg:grid-cols-12 gap-6"
            >
              {/* Left Column File List */}
              <div className="lg:col-span-4 flex flex-col gap-4">
                <div className="bg-[#0b1120] border border-slate-800/80 rounded-2xl p-4 shadow-xl">
                  <div className="px-2 py-1.5 mb-3">
                    <h3 className="text-xs font-bold tracking-wider text-slate-400 uppercase">
                      Workspace Source Files
                    </h3>
                  </div>

                  <div className="flex flex-col gap-1">
                    {pythonFiles.map((file) => (
                      <button
                        key={file.name}
                        onClick={() => {
                          setSelectedFile(file);
                          setCopied(false);
                        }}
                        className={`w-full text-left p-3.5 rounded-xl flex items-start gap-3 transition-all border ${
                          selectedFile.name === file.name
                            ? "bg-slate-900/80 border-cyan-500/30 text-white shadow-sm"
                            : "bg-transparent border-transparent text-slate-400 hover:bg-slate-900/30 hover:text-slate-200"
                        }`}
                      >
                        <FileCode className={`w-5 h-5 flex-shrink-0 mt-0.5 ${
                          selectedFile.name === file.name ? "text-cyan-400" : "text-slate-500"
                        }`} />
                        <div>
                          <p className="text-xs font-bold font-mono tracking-tight">{file.name}</p>
                          <p className="text-[10px] text-slate-500 mt-1 line-clamp-2 leading-relaxed">
                            {file.description}
                          </p>
                        </div>
                      </button>
                    ))}
                  </div>

                  {/* Local download button */}
                  <div className="mt-6 pt-4 border-t border-slate-800/60">
                    <button
                      onClick={() => {
                        pythonFiles.forEach(file => downloadFile(file));
                      }}
                      className="w-full py-3 bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-slate-950 font-bold rounded-xl text-xs flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/10 transition"
                    >
                      <Download className="w-4 h-4 stroke-[2.5]" />
                      Download Complete Python Project (.ZIP)
                    </button>
                    <p className="text-[10px] text-slate-500 font-medium text-center mt-2">
                      Downloads all modules to your local machine as structured python projects.
                    </p>
                  </div>
                </div>

                <div className="bg-slate-900/60 border border-slate-800 p-5 rounded-2xl flex flex-col gap-3">
                  <h4 className="text-xs font-bold text-white flex items-center gap-1.5">
                    <Terminal className="w-4 h-4 text-cyan-400" />
                    How to Run Locally:
                  </h4>
                  <ol className="text-[11px] text-slate-400 space-y-2 list-decimal list-inside leading-relaxed">
                    <li>Create a virtual environment: <br /><code className="text-cyan-400 font-mono px-1 py-0.5 rounded bg-slate-950 text-[10px]">python -m venv venv</code></li>
                    <li>Activate it: <br /><code className="text-cyan-400 font-mono px-1 py-0.5 rounded bg-slate-950 text-[10px]">source venv/bin/activate</code></li>
                    <li>Install packages: <br /><code className="text-cyan-400 font-mono px-1 py-0.5 rounded bg-slate-950 text-[10px]">pip install -r requirements.txt</code></li>
                    <li>Run the main engine: <br /><code className="text-cyan-400 font-mono px-1 py-0.5 rounded bg-slate-950 text-[10px]">python main.py</code></li>
                  </ol>
                </div>
              </div>

              {/* Right Column Code Viewer */}
              <div className="lg:col-span-8 flex flex-col gap-4">
                <div className="bg-[#0b1120] border border-slate-800/80 rounded-2xl overflow-hidden shadow-xl flex flex-col flex-1">
                  
                  {/* File title banner */}
                  <div className="bg-[#0f172a] px-6 py-4 border-b border-slate-800/60 flex items-center justify-between">
                    <div>
                      <h3 className="text-xs font-bold font-mono text-white flex items-center gap-2">
                        {selectedFile.path}
                        <span className="text-[9px] font-sans font-medium bg-slate-800 text-slate-400 px-2 py-0.5 rounded">
                          {selectedFile.name.endsWith(".py") ? "Python script" : "Config file"}
                        </span>
                      </h3>
                      <p className="text-[10px] text-slate-400 mt-1 font-semibold">{selectedFile.description}</p>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={copyToClipboard}
                        className="px-3 py-1.5 bg-slate-800 border border-slate-700 hover:bg-slate-700 text-slate-200 text-xs font-bold rounded-lg flex items-center gap-1.5 transition"
                      >
                        {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                        {copied ? "Copied!" : "Copy Code"}
                      </button>
                      <button
                        onClick={() => downloadFile(selectedFile)}
                        className="p-1.5 bg-slate-800 border border-slate-700 hover:bg-slate-700 text-slate-200 rounded-lg transition"
                        title="Download file"
                      >
                        <Download className="w-4 h-4" />
                      </button>
                    </div>
                  </div>

                  {/* Code Screen */}
                  <div className="p-6 bg-slate-950 overflow-auto max-h-[600px] font-mono text-xs leading-relaxed">
                    <pre className="text-slate-300">
                      <code>{selectedFile.code}</code>
                    </pre>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* TAB 3: GUIDE & ENGINEERING THEORY */}
          {activeTab === "guide" && (
            <motion.div
              key="guide-view"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="bg-[#0b1120] border border-slate-800/80 rounded-2xl p-8 shadow-xl max-w-4xl mx-auto flex flex-col gap-6"
            >
              <div>
                <h2 className="text-xl font-bold text-white mb-2">SORT (Simple Online and Realtime Tracking) Theory</h2>
                <div className="h-0.5 w-20 bg-cyan-500 rounded-full" />
              </div>

              <div className="text-xs text-slate-400 space-y-4 leading-relaxed">
                <p>
                  Object tracking is a core task in computer vision. While object detection identifies bounding boxes in a single frame, object tracking connects those boxes across consecutive frames, assigning unique, persistent identities to targets.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-6">
                  <div className="bg-slate-950 p-5 rounded-xl border border-slate-900/60">
                    <h3 className="text-xs font-bold text-white mb-3 font-mono text-cyan-400">1. KALMAN FILTER STATE SYSTEM</h3>
                    <p className="mb-2">
                      In SORT, the state of each tracking box is modeled via a 2D Kalman Filter with a constant velocity model:
                    </p>
                    <code className="block bg-slate-900 p-3 rounded font-mono text-slate-300 text-[11px] mb-3 leading-loose text-center">
                      x = [u, v, s, r, u_dot, v_dot, s_dot]^T
                    </code>
                    <ul className="list-disc list-inside space-y-1 text-slate-500">
                      <li><code className="text-slate-300 font-mono">u, v</code> : Center pixel coordinate</li>
                      <li><code className="text-slate-300 font-mono">s</code> : Total box area / scale</li>
                      <li><code className="text-slate-300 font-mono">r</code> : Aspect ratio (width / height)</li>
                      <li><code className="text-slate-300 font-mono">u_dot, v_dot, s_dot</code> : Velocity derivatives</li>
                    </ul>
                  </div>

                  <div className="bg-slate-950 p-5 rounded-xl border border-slate-900/60">
                    <h3 className="text-xs font-bold text-white mb-3 font-mono text-cyan-400">2. DATA ASSOCIATION (HUNGARIAN)</h3>
                    <p className="mb-2">
                      Detections in frame $T$ must be paired with predicted tracks. We compute an overlap cost matrix containing the Intersection Over Union (IoU) values:
                    </p>
                    <code className="block bg-slate-900 p-3 rounded font-mono text-slate-300 text-[11px] mb-3 text-center">
                      Cost_ij = -IoU(Detection_i, Predict_j)
                    </code>
                    <p className="text-slate-500">
                      We apply the Hungarian Linear Sum Assignment algorithm. It guarantees the global minimum assignment cost in $O(N^3)$ computational complexity, assuring optimal tracking pairing speeds at 100+ frames per second!
                    </p>
                  </div>
                </div>

                <h3 className="text-sm font-bold text-white">Pruning & Lifecycle Management</h3>
                <p>
                  To manage tracker populations, tracks must handle object exit scenarios or occlusions:
                </p>
                <ul className="list-disc list-inside space-y-2 pl-4 text-slate-400">
                  <li>
                    <strong className="text-slate-300 font-semibold">Min Hits</strong>: When new detections appear, they initialize tracker profiles in a tentative state. If detections are consistently updated for a consecutive number of frames (<code className="text-purple-400 font-mono text-[10px]">min_hits</code>), the tracker becomes fully active. This eliminates sensor flicker noise from creating phantom IDs.
                  </li>
                  <li>
                    <strong className="text-slate-300 font-semibold">Max Age</strong>: When targets go out of sight or get blocked, their frames since update counter increases. We propagate their bounding boxes using the velocity derivatives. If they remain unmatched for longer than (<code className="text-amber-400 font-mono text-[10px]">max_age</code>), the track is deleted from memory, preventing memory leaks and ID swaps.
                  </li>
                </ul>

                <div className="bg-slate-950/60 border border-slate-900 p-5 rounded-xl flex items-start gap-4 mt-6">
                  <AlertCircle className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <h4 className="text-xs font-bold text-white">SORT vs Deep SORT</h4>
                    <p className="text-slate-500 mt-1 leading-relaxed">
                      While standard SORT relies strictly on motion prediction (Kalman filters) and spatial overlaps (IoU), Deep SORT adds a Deep Convolutional Neural Network feature extractor. It extracts a appearance descriptor (cosine feature vector) for every bounding box. This enables object re-identification after very long periods of occlusion, making it extremely robust at the cost of requiring an active GPU to run the feature extraction model.
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/60 bg-[#070a13] py-6 px-6 mt-12 text-center text-xs text-slate-500 font-semibold">
        <p>YOLOv8 & Deep SORT Object Tracking System • Created with React 19, Tailwind CSS and HTML5 Canvas</p>
      </footer>
    </div>
  );
}
