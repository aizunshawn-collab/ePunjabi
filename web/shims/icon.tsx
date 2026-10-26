// Web replacement for 'react-native-vector-icons/MaterialIcons', rendered
// via Google's Material Icons web font (loaded in web/index.html) instead of
// a bundled icon font file. react-native-vector-icons names use hyphens
// (e.g. "graphic-eq"); the web font's ligatures use underscores instead.
import React from 'react';

type IconProps = {
  name: string;
  size?: number;
  color?: string;
  style?: React.CSSProperties;
};

export default function Icon({name, size = 24, color = '#000000', style}: IconProps): JSX.Element {
  return (
    <span
      className="material-icons"
      style={{fontSize: size, color, lineHeight: 1, ...style}}>
      {name.replace(/-/g, '_')}
    </span>
  );
}
