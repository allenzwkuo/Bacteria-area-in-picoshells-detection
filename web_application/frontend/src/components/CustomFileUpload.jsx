import React, { useRef, useState } from 'react';

function CustomFileUpload({ onFileSelect }) {
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);

  const handleButtonClick = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    if (file && onFileSelect) {
      onFileSelect(file);  
    }
  };

  return (
    <div>
      <input
        type="file"
        style={{ display: 'none' }}
        ref={fileInputRef}
        onChange={handleFileChange}
      />
      <button onClick={handleButtonClick} className="file-upload-button">
        {selectedFile ? selectedFile.name : 'Choose File'}
      </button>
    </div>
  );
}

export default CustomFileUpload;
