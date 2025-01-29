import React from 'react';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import WellSelection from "./pages/WellSelection";
import DataUpload from "./pages/DataUpload";
import ModelResults from "./pages/ModelResults";
import Results from "./pages/Results";

function App() {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/well_selection" element={<WellSelection />} />
                <Route path="/upload" element={<DataUpload />} />
                <Route path="/model_results" element={<ModelResults/>} />
                <Route path="/results" element={<Results />} />
            </Routes>
        </Router>
    );
}

export default App;
