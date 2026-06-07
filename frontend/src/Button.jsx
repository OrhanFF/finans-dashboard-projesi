import React from "react";

function Button({ name, onClick, isActive }) {
  const activeClass = isActive ? 'active' : '';

  return (
    <button
      className={`options_button ${activeClass}`}
      onClick={onClick}
    >
      {name}
    </button>
  );
}

export default Button;