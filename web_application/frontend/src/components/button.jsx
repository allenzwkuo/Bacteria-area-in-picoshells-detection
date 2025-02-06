import React from 'react';
import "../styles/button.css";
const Button = ({ size = 'medium', children, onClick }) => {

    return (

        <button className={`button ${size}`} onClick={onClick}>
            {children}
        </button>
    )
}

export default Button;