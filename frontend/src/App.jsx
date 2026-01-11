import { Routes, Route } from "react-router-dom";
import CHDPredictionApp from "./CHDPredictionApp";

function App() {
  return (
    <Routes>
      <Route path="/" element={<CHDPredictionApp />} />
    </Routes>
  );
}

export default App;
