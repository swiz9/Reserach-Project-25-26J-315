import { Routes, Route } from "react-router-dom";
import HeartFailurePredictor from "./HeartFailurePredictor";

function App() {
  return (
    <Routes>
      <Route path="/" element={<HeartFailurePredictor />} />
    </Routes>
  );
}

export default App;
