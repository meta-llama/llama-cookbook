/* eslint-disable no-extra-semi */
/* eslint-disable react/prop-types */
/* eslint-disable no-unused-vars */
import { Route, BrowserRouter as Router, Routes } from "react-router-dom";

import Home from "./pages/homePage";
import BookPage from "./pages/bookPage";

const AppRouter = function () {
  return (
    <>
      <Routes>
        <Route exact path="/" element={<Home />} />
        <Route exact path="/search" element={<BookPage />} />
      </Routes>
    </>
  );
};

const App = () => (
  <Router>
    <AppRouter />
  </Router>
);

export default App;
