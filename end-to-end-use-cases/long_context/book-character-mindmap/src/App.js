import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./homePage/index";
import SearchPage from "./bookPage/components/SearchPage";

function App() {
  return (
    <Router>
      <Routes>
        {/* Define routes for Home and SearchPage */}
        <Route path="/" element={<Home />} />
        <Route path="/search" element={<SearchPage />} />
      </Routes>
    </Router>
  );
}

export default App;
