import { Routes, Route } from "react-router-dom";
import LVHDetectionApp from "./LVHDetectionApp";

function App() {
  return (
    <Routes>
      <Route path="/" element={<LVHDetectionApp />} />
    </Routes>
  );
}

export default App;
