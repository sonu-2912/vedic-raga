import { Camera, ImageUp, RefreshCcw } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

export default function CameraCapture({ onImageReady }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [previewUrl, setPreviewUrl] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, []);

  async function startCamera() {
    setError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraReady(true);
    } catch {
      setCameraReady(false);
      setError('Camera unavailable. Upload an image instead.');
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }

  function handleCapture() {
    const video = videoRef.current;
    if (!video) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 960;
    canvas.height = video.videoHeight || 720;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
      setPreviewUrl(URL.createObjectURL(file));
      onImageReady(file);
    }, 'image/jpeg', 0.92);
  }

  function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setPreviewUrl(URL.createObjectURL(file));
    onImageReady(file);
  }

  return (
    <div className="capture-panel">
      <div className="camera-stage">
        {previewUrl ? (
          <img className="preview-image" src={previewUrl} alt="Selected preview" />
        ) : (
          <video ref={videoRef} className="camera-video" autoPlay muted playsInline />
        )}
      </div>

      {error ? <p className="inline-warning">{error}</p> : null}

      <div className="capture-actions">
        <button className="primary-button" type="button" onClick={handleCapture} disabled={!cameraReady}>
          <Camera size={18} />
          Capture
        </button>
        <label className="secondary-button file-button">
          <ImageUp size={18} />
          Upload
          <input type="file" accept="image/*" onChange={handleUpload} />
        </label>
        <button className="ghost-button" type="button" onClick={() => { setPreviewUrl(''); startCamera(); }}>
          <RefreshCcw size={18} />
          Retake
        </button>
      </div>
    </div>
  );
}

