import React, { useState } from 'react';
import axios from 'axios';
import './styles/styles.css';

const App = () => {
const rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
const cols = Array.from({ length: 12 }, (_, i) => i + 1);
const [selectedWell, setSelectedWell] = useState('');
const [time, setTime] = useState('');
const [concentration, setConcentration] = useState('');
const [percentage, setPercentage] = useState(null);
const [wellData, setWellData] = useState({});
const [results, setResults] = useState({});

const handleWellClick = (well) => {
    setSelectedWell(well);
    setTime('');
    setConcentration('');
    setPercentage(null);
};

const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append('image', file);
    formData.append('well', selectedWell);

    try {
    const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: {
        'Content-Type': 'multipart/form-data',
        },
    });
    setPercentage(response.data.percentage);
    } catch (error) {
    console.error('Error uploading image:', error);
    }
};

const handleSave = () => {
    if (selectedWell && time && concentration && percentage !== null) {
    const newEntry = { time, concentration, percentage };
    setWellData(prev => ({
        ...prev,
        [selectedWell]: [...(prev[selectedWell] || []), newEntry],
    }));
    setSelectedWell('');
    setTime('');
    setConcentration('');
    setPercentage(null);
    }
};

const handleViewResults = async () => {
    try {
    const response = await axios.post('http://localhost:5000/results', { wellData });
    setResults(response.data);
    } catch (error) {
    console.error('Error fetching results:', error);
    }
};

const handleReset = () => {
    setSelectedWell('');
    setTime('');
    setConcentration('');
    setPercentage(null);
    setWellData({});
    setResults({});
}

return (
    <div className="app-container">
    <div className="left-section">
        <h2>Selected Well: {selectedWell}</h2>
        {selectedWell && (
        <>
            <input 
            type="number" 
            placeholder="Time Point" 
            value={time} 
            onChange={(e) => setTime(e.target.value)} 
            />
            <input 
            type="text" 
            placeholder="Concentration of Ampicillin" 
            value={concentration} 
            onChange={(e) => setConcentration(e.target.value)} 
            />
            <input type="file" onChange={handleImageUpload} />
            {percentage !== null && <p>Percentage: {percentage}%</p>}
            <button onClick={handleSave}>Save</button>
        </>
        )}
    </div>

    <div className="right-section">
        {rows.map(row => (
        <div key={row} className="row">
            {cols.map(col => {
            const well = `${row}${col}`;
            return (
                <div 
                key={well} 
                className={`well ${results[well] ? results[well] : ''}`} 
                onClick={() => handleWellClick(well)}
                >
                {well}
                </div>
            );
            })}
        </div>
        ))}
    </div>

    <div className="bottom-section">
        <button onClick={handleViewResults}>View Results</button>
        <button onClick={handleReset} className="reset-button">Reset</button>
    </div>
    </div>
);
};

export default App;
