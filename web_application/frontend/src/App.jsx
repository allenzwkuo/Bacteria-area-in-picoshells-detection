import React, { useState } from 'react';
import "./styles/styles.css";
import Button from "./components/button.jsx";
import image1 from "./images/6_wellplate_image.png";
import image2 from "./images/96_wellplate_image.png";
import logo from "./images/picopals_logo.png";

function App() {

    const [plateType, setPlateType] = useState(null);
    const [selectedWells, setSelectedWells] = useState(new Set());
    const [lockedWells, setLockedWells] = useState(null);
    const [imageUploads, setImageUploads] = useState({});

    const plateLayouts = {
        "96-well": {rows: 8, cols: 12},
        "6-well": {rows: 2, cols: 3},
    };

    const toggleWell = (well) => {
        if (lockedWells) return;
        setSelectedWells((prevSelected) => {
            const newSelection = new Set(prevSelected);
            if (newSelection.has(well)) {
                newSelection.delete(well);
            } else {
                newSelection.add(well);
            }
            return newSelection;
        });
    };

    const lockSelection = () => {
        setLockedWells(new Set(selectedWells));
        setSelectedWells(new Set()); 
    };

    const uploadImage = (well, file) => {
        setImageUploads((prevUploads) => ({
            ...prevUploads,
            [well]: file, 
        }));
    };

    return (
        <div className="container">
            <img src={logo} alt="Logo" className="logo"/>
            {!plateType ? (
                <div className="card">
                    <h1 className="title">Select Plate Type</h1>
                    <div className="button-container">
                        <div className="well-plate-6-button-container">
                            <img className="well-plate-6-image" src={image1} alt="image of a 6 well plate"/>
                            <Button size="large" onClick={() => setPlateType("6-well")}>
                                6-Well Plate
                            </Button>
                        </div>
                        <div className="well-plate-96-button-container">
                            <img className="well-plate-96-image" src={image2} alt="image of a 96 well plate"/>
                            <Button size="large" onClick={() => setPlateType("96-well")}>
                                96-Well Plate
                            </Button>
                        </div>
                    </div>
                </div>
            ) : (
                <div className="card">
                    <h2 className="title">Plate Type: {plateType}</h2>
                    <div 
                        className={`plate-grid ${lockedWells ? "locked-wells" : ""} ${plateType === "6-well" ? "plate-grid-6-well" : "plate-grid-96-well"}`} 
                        style={{
                            display: "grid",
                            gridTemplateColumns: `repeat(${plateLayouts[plateType].cols}, 1fr)`,
                            gridTemplateRows: `repeat(${plateLayouts[plateType].rows}, 1fr)`,
                            gap: "10px",
                            marginTop: "1vh"
                        }}
                    >
                        {[...Array(plateLayouts[plateType].rows * plateLayouts[plateType].cols)].map((_, index) => {
                            const wellId = `Well-${index + 1}`;
                            return (
                                <div
                                    key={wellId}
                                    className={`well ${lockedWells?.has(wellId) ? "locked" : selectedWells.has(wellId) ? "selected" : ""}`}
                                    onClick={() => toggleWell(wellId)}
                                >
                                    {lockedWells?.has(wellId) && (
                                        <input 
                                            type="file"
                                            onChange={(e) => uploadImage(wellId, e.target.files[0])}
                                        />
                                    )}
                                </div>
                            );
                        })}
                    </div>
                    {!lockedWells ? (
                        <div className="continue-button-container">
                            <Button size="medium" onClick={lockSelection} disabled={selectedWells.size === 0}>
                                Continue
                            </Button>
                        </div>
                    ) : (
                        <div className="change-selection-button-container">
                            <Button size="medium" onClick={() => { setPlateType(null); setLockedWells(null); setImageUploads({}); }}>
                                Change Selection
                            </Button>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export default App;
